import os

import FWCore.ParameterSet.Config as cms


def simFromHDF5(process, fileName=None, beamspot=None):
    """Feed a SIM job from a GenHDF5 file (default $GENHDF5_FILE, else gen.h5).

    Event N is row N-1, so a job split by EmptySource firstEvent reads its own rows.
    A file whose vertices carry no beamspot (attribute beamspot=none) is smeared with
    the VtxSmeared of `beamspot` (default $GENHDF5_BEAMSPOT, else DBrealistic).
    A URL (root://...) is staged in by the producer; it is not opened here, so -n is
    required and $GENHDF5_SMEAR=1 asks for the beamspot.
    """
    if fileName is None:
        fileName = os.environ.get("GENHDF5_FILE", "gen.h5")
    if "://" in fileName:
        nEvents = process.maxEvents.input.value()
        if nEvents < 0:
            raise RuntimeError("simFromHDF5: set -n for a remote file " + fileName)
        smeared = os.environ.get("GENHDF5_SMEAR", "0") != "1"
        storedJets = False
        allFinalState = os.environ.get("GENHDF5_PPS", "0") == "1"
    else:
        import h5py
        with h5py.File(fileName.replace("file:", "", 1), "r") as h5:
            nEvents = int(h5["sim/n"].shape[0])
            smeared = h5.attrs.get("beamspot", b"applied") != b"none"
            storedJets = "jets" in h5 and "event/GenMET_pt" in h5
            allFinalState = h5.attrs.get("final_state", b"") == b"all"

    # EmptySource only numbers the events, the producer reads row N
    process.source = cms.Source("EmptySource",
                                firstRun=cms.untracked.uint32(1),
                                firstLuminosityBlock=cms.untracked.uint32(1),
                                firstEvent=cms.untracked.uint32(1),
                                numberEventsInLuminosityBlock=cms.untracked.uint32(nEvents))
    # keep -n, but never run past the end of the file
    if not 0 <= process.maxEvents.input.value() <= nEvents:
        process.maxEvents.input = nEvents

    process.genHDF5Producer = cms.EDProducer("GenHDF5Producer", fileName=cms.string(str(fileName)))

    def alias(cppType, fromInstance=""):
        pset = cms.PSet(type=cms.string(cppType))
        if fromInstance:
            pset.fromProductInstance = cms.string(fromInstance)
            pset.toProductInstance = cms.string("")
        return cms.EDAlias(genHDF5Producer=cms.VPSet(pset))

    # downstream steps find the products under their usual labels
    process.generator = alias("GenEventInfoProduct")
    genTask = cms.Task(process.genHDF5Producer)
    # DIGI in this step (FastSim): drop the DIGI rebuild of what the file provides
    rebuilt = ["genParticles"] + (["ak4GenJetsNoNu", "ak8GenJetsNoNu", "genMetTrue"] if storedJets else [])
    for name in rebuilt:
        if isinstance(getattr(process, name, None), cms.EDProducer):
            delattr(process, name)
    if storedJets:
        process.ak4GenJetsNoNu = alias("recoGenJets", "ak4GenJetsNoNu")
        process.ak8GenJetsNoNu = alias("recoGenJets", "ak8GenJetsNoNu")
        process.genMetTrue = alias("recoGenMETs")
    else:
        # GenJets and GenMET from the stored final state, as the GEN step makes them (genJetMET)
        for cff in ("RecoJets.Configuration.GenJetParticles_cff", "RecoJets.Configuration.RecoGenJets_cff",
                    "RecoMET.Configuration.GenMETParticles_cff", "RecoMET.Configuration.RecoGenMET_cff"):
            process.load(cff)
        genTask.add(process.genJetParticlesTask, process.genMETParticlesTask, process.recoGenJetsTask,
                    process.recoGenMETTask)
    hepmcLabel = "genHDF5Producer"
    if not smeared:
        from Configuration.StandardSequences.VtxSmeared import VtxSmeared
        process.load(VtxSmeared[beamspot or os.environ.get("GENHDF5_BEAMSPOT", "DBrealistic")])
        process.VtxSmeared.src = cms.InputTag("genHDF5Producer")
        hepmcLabel = "VtxSmeared"
    if isinstance(getattr(process, "generatorSmeared", None), cms.EDProducer):
        # SIM and DIGI in one step (FastSim): the DIGI task already copies the HepMC
        process.generatorSmeared.currentTag = cms.untracked.InputTag(hepmcLabel)
    else:
        process.generatorSmeared = cms.EDAlias(
            **{hepmcLabel: cms.VPSet(cms.PSet(type=cms.string("edmHepMCProduct")))})

    # genParticles: the stored truth from the producer, the event origin (xyz0, t0) from the HepMC SIM gets
    from PhysicsTools.HepMCCandAlgos.genParticles_cfi import genParticles as _genParticles
    process.genHDF5Origin = _genParticles.clone(src=hepmcLabel)
    genTask.add(process.genHDF5Origin)
    process.genParticles = cms.EDAlias(
        genHDF5Producer=cms.VPSet(cms.PSet(type=cms.string("recoGenParticles")), cms.PSet(type=cms.string("ints"))),
        genHDF5Origin=cms.VPSet(
            cms.PSet(type=cms.string("floatROOTMathCartesian3DROOTMathDefaultCoordinateSystemTagROOTMathPositionVector3D"),
                     fromProductInstance=cms.string("xyz0"), toProductInstance=cms.string("xyz0")),
            cms.PSet(type=cms.string("float"), fromProductInstance=cms.string("t0"), toProductInstance=cms.string("t0"))))

    if hasattr(process, "g4SimHits"):
        process.g4SimHits.Generator.HepMCProductLabel = cms.InputTag(hepmcLabel)
        # forward protons for PPS: transport them as the GEN step does (era dependent), if the file has them
        process.load("SimPPS.Configuration.GenPPS_cff")
        pps = allFinalState and process.g4SimHits.LHCTransport.value() and len(process.PPSTransportTask.moduleNames()) > 0
        process.g4SimHits.LHCTransport = cms.bool(pps)
        if pps:
            process.LHCTransport.HepMCProductLabel = cms.InputTag(hepmcLabel)
            genTask.add(process.PPSTransportTask)

    for path in process.paths_().values():
        if not smeared:
            path.insert(0, process.VtxSmeared)
        path.insert(0, process.genHDF5Producer)
        path.associate(genTask)

    # GenFilterInfo per lumi, as the GEN step writes it (NanoAOD genFilterTable needs it)
    simPath = next((n for n, p in process.paths_().items() if "g4SimHits" in p.moduleNames()), None)
    if simPath is None and process.schedule is not None:  # e.g. NANO:@GEN straight off the file
        simPath = next((p.label_() for p in process.schedule if isinstance(p, cms.Path)), None)
    if simPath is not None and not hasattr(process, "genFilterEfficiencyProducer"):
        from GeneratorInterface.Core.genFilterEfficiencyProducer_cfi import genFilterEfficiencyProducer
        process.genFilterEfficiencyProducer = genFilterEfficiencyProducer.clone(filterPath=simPath)
        process.genfiltersummary_step = cms.EndPath(process.genFilterEfficiencyProducer)
        if process.schedule is not None:
            process.schedule.insert(process.schedule.index(getattr(process, simPath)) + 1,
                                    process.genfiltersummary_step)

    for name in process.outputModules_():
        om = getattr(process, name)
        if hasattr(om, "outputCommands"):
            om.outputCommands.append("drop *_genHDF5Producer_*_*")
            om.outputCommands.append("drop *_genHDF5Origin_*_*")
            if not smeared:
                om.outputCommands.append("drop *_VtxSmeared_*_*")

    return process


def simFromHDF5RelVal(process):
    """RelVal input: 100 TTbar_14TeV_TuneCP5 2025 events from the data package."""
    from FWCore.ParameterSet.pfnInPath import pfnInPath

    fileName = pfnInPath("IOMC/Input/data/TTbar_14TeV_TuneCP5_2025.h5")
    return simFromHDF5(process, fileName.replace("file:", "", 1))


def digiFromHDF5(process):
    """DIGI step for h5 input: keep the SIM step's genParticles, GenJets and GenMET.

    The standard DIGI rebuilds them from the HepMC SIM got (fixGenInfoTask), which for h5 input is
    the SIM tier only and would replace the stored truth.
    """
    import re

    # SIM in the same step (FastSim): simFromHDF5 handles it
    if any(hasattr(process, m) for m in ("genHDF5Producer", "g4SimHits", "fastSimProducer")):
        return process
    if hasattr(process, "fixGenInfoTask"):
        for name in process.fixGenInfoTask.moduleNames():
            if hasattr(process, name):
                delattr(process, name)
    # cmsDriver drops the input gen collections it expects the DIGI step to rebuild
    if hasattr(process.source, "inputCommands"):
        gen = re.compile(r"drop \*_(genParticles|genParticlesForJets|\w*GenJets\w*|genCandidatesForMET|genParticlesForMETAllVisible|genMet\w*)_\*_\*$")
        process.source.inputCommands = [c for c in process.source.inputCommands if not gen.match(c.strip())]
    return process
