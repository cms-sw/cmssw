import os

import FWCore.ParameterSet.Config as cms


def simFromHDF5(process, fileName=None):
    """Feed a SIM job from a GenHDF5 file (default $GENHDF5_FILE, else gen.h5)."""
    import h5py

    if fileName is None:
        fileName = os.environ.get("GENHDF5_FILE", "gen.h5")
    with h5py.File(fileName, "r") as h5:
        nEvents = int(h5["sim/n"].shape[0])

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
    process.genParticles = alias("recoGenParticles")
    process.ak4GenJetsNoNu = alias("recoGenJets", "ak4GenJetsNoNu")
    process.ak8GenJetsNoNu = alias("recoGenJets", "ak8GenJetsNoNu")
    process.genMetTrue = alias("recoGenMETs")
    process.generator = alias("GenEventInfoProduct")
    process.generatorSmeared = alias("edmHepMCProduct")

    if hasattr(process, "g4SimHits"):
        process.g4SimHits.Generator.HepMCProductLabel = cms.InputTag("genHDF5Producer")
        process.g4SimHits.LHCTransport = cms.bool(False)

    for path in process.paths_().values():
        path.insert(0, process.genHDF5Producer)

    # GenFilterInfo per lumi, as the GEN step writes it (NanoAOD genFilterTable needs it)
    simPath = next((n for n, p in process.paths_().items() if "g4SimHits" in p.moduleNames()), None)
    if simPath is not None:
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

    return process


def simFromHDF5RelVal(process):
    """RelVal input: 100 TTbar_14TeV_TuneCP5 2025 events from the data package."""
    from FWCore.ParameterSet.pfnInPath import pfnInPath

    fileName = pfnInPath("IOMC/Input/data/TTbar_14TeV_TuneCP5_2025.h5")
    return simFromHDF5(process, fileName.replace("file:", "", 1))
