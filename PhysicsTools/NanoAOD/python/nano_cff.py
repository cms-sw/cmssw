from __future__ import print_function
import FWCore.ParameterSet.Config as cms
from PhysicsTools.NanoAOD.common_cff import *
from PhysicsTools.NanoAOD.nano_eras_cff import *
from PhysicsTools.NanoAOD.jetsAK4_CHS_cff import *
from PhysicsTools.NanoAOD.jetsAK4_Puppi_cff import *
from PhysicsTools.NanoAOD.jetsAK8_cff import *
from PhysicsTools.NanoAOD.jetMC_cff import *
from PhysicsTools.NanoAOD.muons_cff import *
from PhysicsTools.NanoAOD.taus_cff import *
from PhysicsTools.NanoAOD.boostedTaus_cff import *
from PhysicsTools.NanoAOD.electrons_cff import *
from PhysicsTools.NanoAOD.lowPtElectrons_cff import *
from PhysicsTools.NanoAOD.photons_cff import *
from PhysicsTools.NanoAOD.globals_cff import *
from PhysicsTools.NanoAOD.extraflags_cff import *
from PhysicsTools.NanoAOD.ttbarCategorization_cff import *
from PhysicsTools.NanoAOD.genparticles_cff import *
from PhysicsTools.NanoAOD.particlelevel_cff import *
from PhysicsTools.NanoAOD.genWeightsTable_cfi import *
from PhysicsTools.NanoAOD.genVertex_cff import *
from PhysicsTools.NanoAOD.vertices_cff import *
from PhysicsTools.NanoAOD.met_cff import *
from PhysicsTools.NanoAOD.triggerObjects_cff import *
from PhysicsTools.NanoAOD.isotracks_cff import *
from PhysicsTools.NanoAOD.protons_cff import *
from PhysicsTools.NanoAOD.NanoAODEDMEventContent_cff import *
from PhysicsTools.NanoAOD.fsrPhotons_cff import *
from PhysicsTools.NanoAOD.softActivity_cff import *
from PhysicsTools.NanoAOD.l1trig_cff import *

nanoMetadata = cms.EDProducer("UniqueStringProducer",
    strings = cms.PSet(
        tag = cms.string("untagged"),
    )
)

linkedObjects = cms.EDProducer("PATObjectCrossLinker",
   jets=cms.InputTag("finalJetsPuppi"),
   muons=cms.InputTag("finalMuons"),
   electrons=cms.InputTag("finalElectrons"),
   lowPtElectrons=cms.InputTag("finalLowPtElectrons"),
   taus=cms.InputTag("finalTaus"),
   boostedTaus=cms.InputTag("finalBoostedTaus"),
   photons=cms.InputTag("finalPhotons"),
   vertices=cms.InputTag("slimmedSecondaryVertices")
)

# Switch to AK4 CHS jets for Run-2
run2_nanoAOD_ANY.toModify(
    linkedObjects, jets="finalJets"
)

# boosted taus don't exist in 122X MINI
run3_nanoAOD_122.toModify(
    linkedObjects, boostedTaus=None,
)

from PhysicsTools.NanoAOD.lhcInfoProducer_cfi import lhcInfoProducer
lhcInfoTable = lhcInfoProducer.clone()
(~run3_common).toModify(
    lhcInfoTable, useNewLHCInfo=False
)

nanoTableTaskCommon = cms.Task(
    cms.Task(nanoMetadata), 
    jetPuppiTask, jetPuppiForMETTask, jetAK8Task,
    extraFlagsProducersTask, muonTask, tauTask, boostedTauTask,
    electronTask , lowPtElectronTask, photonTask,
    vertexTask, isoTrackTask, jetAK8LepTask,  # must be after all the leptons
    softActivityTask,
    cms.Task(linkedObjects),
    jetPuppiTablesTask, jetAK8TablesTask,
    muonTablesTask, fsrTablesTask, tauTablesTask, boostedTauTablesTask,
    electronTablesTask, lowPtElectronTablesTask, photonTablesTask,
    globalTablesTask, vertexTablesTask, metTablesTask, extraFlagsTableTask,
    isoTrackTablesTask,softActivityTablesTask
)

# Replace AK4 Puppi with AK4 CHS for Run-2
_nanoTableTaskCommonRun2 = nanoTableTaskCommon.copy()
_nanoTableTaskCommonRun2.replace(jetPuppiTask, jetTask)
_nanoTableTaskCommonRun2.replace(jetPuppiForMETTask, jetForMETTask)
_nanoTableTaskCommonRun2.replace(jetPuppiTablesTask, jetTablesTask)
run2_nanoAOD_ANY.toReplaceWith(
    nanoTableTaskCommon, _nanoTableTaskCommonRun2
)

nanoSequenceCommon = cms.Sequence(nanoTableTaskCommon)

nanoSequenceOnlyFullSim = cms.Sequence(triggerObjectTablesTask)
nanoSequenceOnlyData = cms.Sequence(cms.Sequence(protonTablesTask) + lhcInfoTable)

nanoSequence = cms.Sequence(nanoSequenceCommon + nanoSequenceOnlyData + nanoSequenceOnlyFullSim)

nanoTableTaskFS = cms.Task(
    genParticleTask, particleLevelTask, jetMCTask, muonMCTask, electronMCTask, lowPtElectronMCTask, photonMCTask,
    tauMCTask, boostedTauMCTask,
    metMCTable, ttbarCatMCProducersTask, globalTablesMCTask, ttbarCategoryTableTask,
    genWeightsTableTask, genVertexTablesTask, genParticleTablesTask, genProtonTablesTask, particleLevelTablesTask
)

nanoSequenceFS = cms.Sequence(nanoSequenceCommon + cms.Sequence(nanoTableTaskFS))

# GenVertex only stored in newer MiniAOD
nanoSequenceMC = nanoSequenceFS.copy()
nanoSequenceMC.insert(nanoSequenceFS.index(nanoSequenceCommon)+1,nanoSequenceOnlyFullSim)

# modifier which adds new tauIDs (currently only deepTauId2017v2p1 is being added)
import RecoTauTag.RecoTau.tools.runTauIdMVA as tauIdConfig
def nanoAOD_addTauIds(process, idsToRun=[]):
    if idsToRun: #no-empty list of tauIDs to run
        updatedTauName = "slimmedTausUpdated"
        tauIdEmbedder = tauIdConfig.TauIDEmbedder(process, debug = False,
                                                  updatedTauName = updatedTauName,
                                                  postfix = "ForNano",
            toKeep = idsToRun)
        tauIdEmbedder.runTauID()
        process.finalTaus.src = updatedTauName
        #remember to adjust the selection and tables with added IDs

        process.tauTask.add( process.rerunMvaIsolationTaskForNano, getattr(process, updatedTauName) )

    return process

def nanoAOD_addBoostedTauIds(process, idsToRun=[]):
    if idsToRun: #no-empty list of tauIDs to run
        updatedBoostedTauName = "slimmedTausBoostedNewID"
        boostedTauIdEmbedder = tauIdConfig.TauIDEmbedder(process, debug = False,
                                                         originalTauName = "slimmedTausBoosted",
                                                         updatedTauName = updatedBoostedTauName,
                                                         postfix = "BoostedForNano",
                                                         toKeep = idsToRun)
        boostedTauIdEmbedder.runTauID()
        process.finalBoostedTaus.src = updatedBoostedTauName
        #remember to adjust the selection and tables with added IDs

        process.boostedTauTask.add( process.rerunMvaIsolationTaskBoostedForNano, getattr(process, updatedBoostedTauName))

    return process

def nanoAOD_addUTagToTaus(process, addUTagInfo=False, usePUPPIjets=False):
    
    if addUTagInfo:
        originalTauName = process.finalTaus.src.value()
        
        if usePUPPIjets: # option to use PUPPI jets   
            jetCollection = "updatedJetsPuppi"
            TagName = "pfUnifiedParticleTransformerAK4JetTags"
            tag_prefix = "byUTagPUPPI"
            updatedTauName = originalTauName+'WithUTagPUPPI'
            # Unified ParT Tagger used for PUPPI jets
            from RecoBTag.ONNXRuntime.pfUnifiedParticleTransformerAK4JetTags_cfi import pfUnifiedParticleTransformerAK4JetTags
            Discriminators = [TagName+":"+tag for tag in pfUnifiedParticleTransformerAK4JetTags.flav_names.value()]
        else: # use CHS jets by default
            jetCollection = "updatedJets"
            TagName = "pfParticleNetFromMiniAODAK4CHSCentralJetTags"
            tag_prefix = "byUTagCHS"
            updatedTauName = originalTauName+'WithUTagCHS'
            # PNet tagger used for CHS jets
            from RecoBTag.ONNXRuntime.pfParticleNetFromMiniAODAK4_cff import pfParticleNetFromMiniAODAK4CHSCentralJetTags
            Discriminators = [TagName+":"+tag for tag in pfParticleNetFromMiniAODAK4CHSCentralJetTags.flav_names.value()]

        # Define "hybridTau" producer
        from PhysicsTools.PatAlgos.patTauHybridProducer_cfi import patTauHybridProducer
        setattr(process, updatedTauName, patTauHybridProducer.clone(
            src = originalTauName,
            jetSource = jetCollection,
            dRMax = 0.4,
            jetPtMin = 15,
            jetEtaMax = 2.5,
            UTagLabel = TagName,
            UTagScoreNames = Discriminators,
            tagPrefix = tag_prefix,
            tauScoreMin = -1,
            vsJetMin = 0.05,
            checkTauScoreIsBest = False,
            chargeAssignmentProbMin = 0.2,
            addGenJetMatch = False,
            genJetMatch = ""
        ))
        process.finalTaus.src = updatedTauName

        #remember to adjust the selection and tables with added IDs

        process.tauTask.add(process.jetTask, getattr(process, updatedTauName))

    return process

from PhysicsTools.SelectorUtils.tools.vid_id_tools import *
def nanoAOD_activateVID(process):

    switchOnVIDElectronIdProducer(process,DataFormat.MiniAOD,electronTask)
    for modname in electron_id_modules_WorkingPoints_nanoAOD.modules:
        setupAllVIDIdsInModule(process,modname,setupVIDElectronSelection)

    process.electronTask.add( process.egmGsfElectronIDTask )

    switchOnVIDPhotonIdProducer(process,DataFormat.MiniAOD,photonTask) # do not call this to avoid resetting photon IDs in VID, if called before inside makePuppiesFromMiniAOD
    for modname in photon_id_modules_WorkingPoints_nanoAOD.modules:
        setupAllVIDIdsInModule(process,modname,setupVIDPhotonSelection)

    process.photonTask.add( process.egmPhotonIDTask )

    return process

def nanoAOD_customizeCommon(process):

    process = nanoAOD_activateVID(process)

    run2_nanoAOD_106Xv2.toModify(
        nanoAOD_addDeepInfoAK4CHS_switch, nanoAOD_addParticleNet_switch=True,
        nanoAOD_addRobustParTAK4Tag_switch=False,
        nanoAOD_addUnifiedParTAK4Tag_switch=True,
    )
  
    # enable rerun of PNet for CHS jets for early run3 eras
    # (it is rerun for run2 within jet tasks while is not needed for newer
    # run3 eras as it is present in miniAOD)
    (run3_nanoAOD_122 | run3_nanoAOD_124).toModify(
        nanoAOD_addDeepInfoAK4CHS_switch, nanoAOD_addParticleNet_switch = True
    )
    
    # This function is defined in jetsAK4_Puppi_cff.py
    process = nanoAOD_addDeepInfoAK4(process,
        addParticleNet=nanoAOD_addDeepInfoAK4_switch.nanoAOD_addParticleNet_switch,
        addRobustParTAK4=nanoAOD_addDeepInfoAK4_switch.nanoAOD_addRobustParTAK4Tag_switch,
        addUnifiedParTAK4=nanoAOD_addDeepInfoAK4_switch.nanoAOD_addUnifiedParTAK4Tag_switch
    )

    # This function is defined in jetsAK4_CHS_cff.py
    process = nanoAOD_addDeepInfoAK4CHS(process,
        addDeepBTag=nanoAOD_addDeepInfoAK4CHS_switch.nanoAOD_addDeepBTag_switch,
        addDeepFlavour=nanoAOD_addDeepInfoAK4CHS_switch.nanoAOD_addDeepFlavourTag_switch,
        addParticleNet=nanoAOD_addDeepInfoAK4CHS_switch.nanoAOD_addParticleNet_switch,
        addRobustParTAK4=nanoAOD_addDeepInfoAK4CHS_switch.nanoAOD_addRobustParTAK4Tag_switch,
        addUnifiedParTAK4=nanoAOD_addDeepInfoAK4CHS_switch.nanoAOD_addUnifiedParTAK4Tag_switch
    )

    # This function is defined in jetsAK8_cff.py
    process = nanoAOD_addDeepInfoAK8(process,
        addDeepBTag=nanoAOD_addDeepInfoAK8_switch.nanoAOD_addDeepBTag_switch,
        addDeepBoostedJet=nanoAOD_addDeepInfoAK8_switch.nanoAOD_addDeepBoostedJet_switch,
        addDeepDoubleX=nanoAOD_addDeepInfoAK8_switch.nanoAOD_addDeepDoubleX_switch,
        addDeepDoubleXV2=nanoAOD_addDeepInfoAK8_switch.nanoAOD_addDeepDoubleXV2_switch,
        addParticleNetMassLegacy=nanoAOD_addDeepInfoAK8_switch.nanoAOD_addParticleNetMassLegacy_switch,
        addParticleNet=nanoAOD_addDeepInfoAK8_switch.nanoAOD_addParticleNet_switch,
        jecPayload=nanoAOD_addDeepInfoAK8_switch.jecPayload
    )

    nanoAOD_tau_switch = cms.PSet(
        idsToAdd = cms.vstring(),
        addUParTInfo = cms.bool(True),
        addPNet = cms.bool(True)
    )
    (run2_nanoAOD_106Xv2 | run3_nanoAOD_122).toModify(
        nanoAOD_tau_switch, idsToAdd = ["deepTau2018v2p5"]
    ).toModify(
        process, lambda p : nanoAOD_addTauIds(p, nanoAOD_tau_switch.idsToAdd.value())
    )
    
    # Don't add Unified Tagger for PUPPI jets for Run 2 (as different PUPPI tune
    # and base jet algorithm) or early Run 3 eras
    (run3_nanoAOD_122 | run3_nanoAOD_124 | run2_nanoAOD_106Xv2).toModify(
        nanoAOD_tau_switch, addUParTInfo = False
    )
    
    # Add Unified Tagger For CHS Jets (PNet 2023)
    nanoAOD_addUTagToTaus(process,
                          addUTagInfo = nanoAOD_tau_switch.addPNet.value(),
                          usePUPPIjets = False
    )

    # Add Unified Tagger For PUPPI Jets (UParT 2024)
    nanoAOD_addUTagToTaus(process,
                        addUTagInfo = nanoAOD_tau_switch.addUParTInfo.value(),
                        usePUPPIjets = True
    )
    
    nanoAOD_boostedTau_switch = cms.PSet(
        idsToAdd = cms.vstring()
    )
    run2_nanoAOD_106Xv2.toModify(
        nanoAOD_boostedTau_switch, idsToAdd = ["mvaIso", "mvaIsoNewDM", "mvaIsoDR0p3", "againstEle"]
    ).toModify(
        process, lambda p : nanoAOD_addBoostedTauIds(p, nanoAOD_boostedTau_switch.idsToAdd.value())
    )

    # Add lepton time-life info
    from PhysicsTools.NanoAOD.leptonTimeLifeInfo_common_cff import addTimeLifeInfoBase
    process = addTimeLifeInfoBase(process)
    
    process = nanoAOD_refineFastSim_puppiJet(process)
    process = nanoAOD_refineFastSim_bTagDeepFlav(process)

    return process

###increasing the precision of selected GenParticles.
def nanoWmassGenCustomize(process):
    pdgSelection="?(abs(pdgId) == 11|| abs(pdgId)==13 || abs(pdgId)==15 ||abs(pdgId)== 12 || abs(pdgId)== 14 || abs(pdgId)== 16|| abs(pdgId)== 24|| pdgId== 23)"
    # Keep precision same as default RECO for selected particles
    ptPrecision="{}?{}:{}".format(pdgSelection, CandVars.pt.precision.value(),genParticleTable.variables.pt.precision.value())
    process.genParticleTable.variables.pt.precision=cms.string(ptPrecision)
    phiPrecision="{} ? {} : {}".format(pdgSelection, CandVars.phi.precision.value(), genParticleTable.variables.phi.precision.value())
    process.genParticleTable.variables.phi.precision=cms.string(phiPrecision)
    etaPrecision="{} ? {} : {}".format(pdgSelection, CandVars.eta.precision.value(), genParticleTable.variables.eta.precision.value())
    process.genParticleTable.variables.eta.precision=cms.string(etaPrecision)
    return process

def nanoL1TrigObjCustomize(process):
    process.nanoTableTaskCommon.add(process.l1TablesTask)
    process = setL1NanoToReduced(process)
    return process

def nanoL1TrigObjCustomizeFull(process):
    process.nanoTableTaskCommon.add(process.l1TablesTask)
    return process
