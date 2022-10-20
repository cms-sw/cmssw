from __future__ import print_function
import FWCore.ParameterSet.Config as cms
##these imports do not need to be copied, as they are very common definitions
from PhysicsTools.NanoAOD.common_cff import *
from PhysicsTools.NanoAOD.nano_eras_cff import *
from PhysicsTools.NanoAOD.NanoAODEDMEventContent_cff import *
##these imports do need to be "frozen" for V10
from PhysicsTools.NanoAOD.V10.jetsAK4_CHS_cff import *
from PhysicsTools.NanoAOD.V10.jetsAK4_Puppi_cff import *
from PhysicsTools.NanoAOD.V10.jetsAK8_cff import *
from PhysicsTools.NanoAOD.V10.jetMC_cff import *
from PhysicsTools.NanoAOD.V10.muons_cff import *
from PhysicsTools.NanoAOD.V10.taus_cff import *
from PhysicsTools.NanoAOD.V10.boostedTaus_cff import *
from PhysicsTools.NanoAOD.V10.electrons_cff import *
from PhysicsTools.NanoAOD.V10.lowPtElectrons_cff import *
from PhysicsTools.NanoAOD.V10.photons_cff import *
from PhysicsTools.NanoAOD.V10.globals_cff import *
from PhysicsTools.NanoAOD.V10.extraflags_cff import *
from PhysicsTools.NanoAOD.V10.ttbarCategorization_cff import *
from PhysicsTools.NanoAOD.V10.genparticles_cff import *
from PhysicsTools.NanoAOD.V10.particlelevel_cff import *
from PhysicsTools.NanoAOD.V10.genWeightsTable_cfi import *
from PhysicsTools.NanoAOD.V10.genVertex_cff import *
from PhysicsTools.NanoAOD.V10.vertices_cff import *
from PhysicsTools.NanoAOD.V10.met_cff import *
from PhysicsTools.NanoAOD.V10.triggerObjects_cff import *
from PhysicsTools.NanoAOD.V10.isotracks_cff import *
from PhysicsTools.NanoAOD.V10.protons_cff import *
from PhysicsTools.NanoAOD.V10.btagWeightTable_cff import *
from PhysicsTools.NanoAOD.V10.fsrPhotons_cff import *
from PhysicsTools.NanoAOD.V10.softActivity_cff import *

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
   photons=cms.InputTag("finalPhotons"),
)

# Switch to AK4 CHS jets for Run-2
run2_nanoAOD_ANY.toModify(linkedObjects, jets="finalJets")

simpleCleanerTable = cms.EDProducer("NanoAODSimpleCrossCleaner",
   name=cms.string("cleanmask"),
   doc=cms.string("simple cleaning mask with priority to leptons"),
   jets=cms.InputTag("linkedObjects","jets"),
   muons=cms.InputTag("linkedObjects","muons"),
   electrons=cms.InputTag("linkedObjects","electrons"),
   lowPtElectrons=cms.InputTag("linkedObjects","lowPtElectrons"),
   taus=cms.InputTag("linkedObjects","taus"),
   photons=cms.InputTag("linkedObjects","photons"),
   jetSel=cms.string("pt>15"),
   muonSel=cms.string("track.isNonnull && isLooseMuon && isPFMuon && innerTrack.validFraction >= 0.49 && ( isGlobalMuon && globalTrack.normalizedChi2 < 3 && combinedQuality.chi2LocalPosition < 12 && combinedQuality.trkKink < 20 && segmentCompatibility >= 0.303 || segmentCompatibility >= 0.451 )"),
   electronSel=cms.string(""),
   lowPtElectronSel=cms.string(""),
   tauSel=cms.string(""),
   photonSel=cms.string(""),
   jetName=cms.string("Jet"),muonName=cms.string("Muon"),electronName=cms.string("Electron"),
   lowPtElectronName=cms.string("LowPtElectron"),
   tauName=cms.string("Tau"),photonName=cms.string("Photon")
)


lhcInfoTable = cms.EDProducer("LHCInfoProducer",
    precision = cms.int32(10),
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
    globalTablesTask, vertexTablesTask, metTablesTask, simpleCleanerTable, extraFlagsTableTask,
    isoTrackTablesTask,softActivityTablesTask
)

# Replace AK4 Puppi with AK4 CHS for Run-2
_nanoTableTaskCommonRun2 = nanoTableTaskCommon.copy()
_nanoTableTaskCommonRun2.replace(jetPuppiTask, jetTask)
_nanoTableTaskCommonRun2.replace(jetPuppiForMETTask, jetForMETTask)
_nanoTableTaskCommonRun2.replace(jetPuppiTablesTask, jetTablesTask)
run2_nanoAOD_ANY.toReplaceWith(nanoTableTaskCommon, _nanoTableTaskCommonRun2)

nanoSequenceCommon = cms.Sequence(nanoTableTaskCommon)

nanoSequenceOnlyFullSim = cms.Sequence(triggerObjectTablesTask)
nanoSequenceOnlyData = cms.Sequence(cms.Sequence(protonTablesTask) + lhcInfoTable)

nanoSequence = cms.Sequence(nanoSequenceCommon + nanoSequenceOnlyData + nanoSequenceOnlyFullSim)

nanoTableTaskFS = cms.Task(
    genParticleTask, particleLevelTask, jetMCTask, muonMCTask, electronMCTask, lowPtElectronMCTask, photonMCTask,
    tauMCTask, boostedTauMCTask,
    metMCTable, ttbarCatMCProducersTask, globalTablesMCTask, cms.Task(btagWeightTable), ttbarCategoryTableTask,
    genWeightsTableTask, genVertexTablesTask, genParticleTablesTask, genProtonTablesTask, particleLevelTablesTask
)

nanoSequenceFS = cms.Sequence(nanoSequenceCommon + cms.Sequence(nanoTableTaskFS))

# GenVertex only stored in newer MiniAOD
nanoSequenceMC = nanoSequenceFS.copy()
nanoSequenceMC.insert(nanoSequenceFS.index(nanoSequenceCommon)+1,nanoSequenceOnlyFullSim)

# modifier which adds new tauIDs (currently only deepTauId2017v2p1 is being added)
## might be an issue with V10 support in newer release
import RecoTauTag.RecoTau.tools.runTauIdMVA as tauIdConfig
def nanoAOD_addTauIds(process, idsToRun=[]):
    if idsToRun: #no-empty list of tauIDs to run
        updatedTauName = "slimmedTausUpdated"
        tauIdEmbedder = tauIdConfig.TauIDEmbedder(process, debug = False,
                                                  updatedTauName = updatedTauName,
                                                  postfix = "ForNano",
            toKeep = idsToRun)
        tauIdEmbedder.runTauID()
        _tauTask = patTauMVAIDsTask.copy()
        _tauTask.add(process.rerunMvaIsolationTaskForNano)
        _tauTask.add(finalTaus)
        process.finalTaus.src = updatedTauName
        #remember to adjust the selection and tables with added IDs

        process.tauTask = _tauTask.copy()

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
        _boostedTauTask = process.rerunMvaIsolationTaskBoostedForNano.copy()
        _boostedTauTask.add(getattr(process, updatedBoostedTauName))
        _boostedTauTask.add(process.finalBoostedTaus)
        process.finalBoostedTaus.src = updatedBoostedTauName
        #remember to adjust the selection and tables with added IDs

        process.boostedTauTask = _boostedTauTask.copy()

    return process

from PhysicsTools.PatUtils.tools.runMETCorrectionsAndUncertainties import runMetCorAndUncFromMiniAOD
def nanoAOD_recalibrateMETs(process,isData):
    # add DeepMETs
    nanoAOD_DeepMET_switch = cms.PSet(
        ResponseTune_Graph = cms.untracked.string('RecoMET/METPUSubtraction/data/models/deepmet/deepmet_resp_v1_2018/model.graphdef')
    )

    print("add DeepMET Producers")
    process.load('RecoMET.METPUSubtraction.deepMETProducer_cfi')
    process.deepMETsResolutionTune = process.deepMETProducer.clone()
    process.deepMETsResponseTune = process.deepMETProducer.clone()
    process.deepMETsResponseTune.graph_path = nanoAOD_DeepMET_switch.ResponseTune_Graph.value()

    runMetCorAndUncFromMiniAOD(process,isData=isData)
    process.nanoSequenceCommon.insert(2,cms.Sequence(process.fullPatMetSequence))


    from PhysicsTools.PatAlgos.slimming.puppiForMET_cff import makePuppiesFromMiniAOD
    makePuppiesFromMiniAOD(process,True)
    process.puppiNoLep.useExistingWeights = True
    process.puppi.useExistingWeights = True
    run2_nanoAOD_106Xv1.toModify(process.puppiNoLep, useExistingWeights = False)
    run2_nanoAOD_106Xv1.toModify(process.puppi, useExistingWeights = False)
    print("will make Puppies on top of MINIAOD")

#    makePuppiesFromMiniAOD(process,True) # call this before in the global customizer otherwise it would reset photon IDs in VID
    nanoAOD_PuppiV15_switch = cms.PSet(
            recoMetFromPFCs = cms.untracked.bool(False),
            reclusterJets = cms.untracked.bool(False),
            )
    run2_nanoAOD_106Xv1.toModify(nanoAOD_PuppiV15_switch,recoMetFromPFCs=True,reclusterJets=True)
    if nanoAOD_PuppiV15_switch.reclusterJets:
        from RecoJets.JetProducers.ak4PFJets_cfi import ak4PFJets
        from PhysicsTools.PatAlgos.tools.helpers import getPatAlgosToolsTask, addToProcessAndTask
        task = getPatAlgosToolsTask(process)
        addToProcessAndTask('ak4PuppiJets', ak4PFJets.clone (src = 'puppi', doAreaFastjet = True, jetPtMin = 10.), process, task)
        from PhysicsTools.PatAlgos.tools.jetTools import addJetCollection
        addJetCollection(process,
            labelName = 'Puppi',
            jetSource = cms.InputTag('ak4PuppiJets'),
            algo = 'AK', rParam=0.4,
            genJetCollection=cms.InputTag('slimmedGenJets'),
            jetCorrections = ('AK4PFPuppi', ['L1FastJet', 'L2Relative', 'L3Absolute','L2L3Residual'], 'None'),
            pfCandidates = cms.InputTag('packedPFCandidates'),
            pvSource = cms.InputTag('offlineSlimmedPrimaryVertices'),
            svSource = cms.InputTag('slimmedSecondaryVertices'),
            muSource =cms.InputTag( 'slimmedMuons'),
            elSource = cms.InputTag('slimmedElectrons'),
            genParticles= cms.InputTag('prunedGenParticles'),
            getJetMCFlavour= False
        )

        process.patJetsPuppi.addGenPartonMatch = cms.bool(False)
        process.patJetsPuppi.addGenJetMatch = cms.bool(False)

        print("nanoAOD_PuppiV15_switch.reclusterJets is true")

    runMetCorAndUncFromMiniAOD(process,isData=isData,metType="Puppi",postfix="Puppi",jetFlavor="AK4PFPuppi", recoMetFromPFCs=bool(nanoAOD_PuppiV15_switch.recoMetFromPFCs), reclusterJets=bool(nanoAOD_PuppiV15_switch.reclusterJets))
    process.nanoSequenceCommon.insert(2,cms.Sequence(process.puppiMETSequence+process.fullPatMetSequencePuppi))

    return process

from PhysicsTools.SelectorUtils.tools.vid_id_tools import *
def nanoAOD_activateVID(process):

    switchOnVIDElectronIdProducer(process,DataFormat.MiniAOD,electronTask)
    for modname in electron_id_modules_WorkingPoints_nanoAOD.modules:
        setupAllVIDIdsInModule(process,modname,setupVIDElectronSelection)

    electronTask_ = process.egmGsfElectronIDTask.copy()
    electronTask_.add(electronTask.copy())
    process.electronTask = electronTask_.copy()

    switchOnVIDPhotonIdProducer(process,DataFormat.MiniAOD,photonTask) # do not call this to avoid resetting photon IDs in VID, if called before inside makePuppiesFromMiniAOD
    for modname in photon_id_modules_WorkingPoints_nanoAOD.modules:
        setupAllVIDIdsInModule(process,modname,setupVIDPhotonSelection)

    photonTask_ = process.egmPhotonIDTask.copy()
    photonTask_.add(photonTask.copy())
    process.photonTask = photonTask_.copy()
    return process

from PhysicsTools.PatUtils.tools.runMETCorrectionsAndUncertainties import runMetCorAndUncFromMiniAOD
def nanoAOD_runMETfixEE2017(process,isData):
    runMetCorAndUncFromMiniAOD(process,isData=isData,
                               fixEE2017 = True,
                               fixEE2017Params = {'userawPt': True, 'ptThreshold':50.0, 'minEtaThreshold':2.65, 'maxEtaThreshold': 3.139},
                               postfix = "FixEE2017")
    process.nanoSequenceCommon.insert(2,process.fullPatMetSequenceFixEE2017)


def nanoAOD_customizeCommon(process):

    process = nanoAOD_activateVID(process)

    # This function is defined in jetsAK4_CHS_cff.py
    process = nanoAOD_addDeepInfoAK4CHS(process,
        addDeepBTag=nanoAOD_addDeepInfoAK4CHS_switch.nanoAOD_addDeepBTag_switch,
        addDeepFlavour=nanoAOD_addDeepInfoAK4CHS_switch.nanoAOD_addDeepFlavourTag_switch
    )

    # This function is defined in jetsAK8_cff.py
    process = nanoAOD_addDeepInfoAK8(process,
        addDeepBTag=nanoAOD_addDeepInfoAK8_switch.nanoAOD_addDeepBTag_switch,
        addDeepBoostedJet=nanoAOD_addDeepInfoAK8_switch.nanoAOD_addDeepBoostedJet_switch,
        addDeepDoubleX=nanoAOD_addDeepInfoAK8_switch.nanoAOD_addDeepDoubleX_switch,
        addDeepDoubleXV2=nanoAOD_addDeepInfoAK8_switch.nanoAOD_addDeepDoubleXV2_switch,
        addParticleNet=nanoAOD_addDeepInfoAK8_switch.nanoAOD_addParticleNet_switch,
        addParticleNetMass=nanoAOD_addDeepInfoAK8_switch.nanoAOD_addParticleNetMass_switch,
        jecPayload=nanoAOD_addDeepInfoAK8_switch.jecPayload
    )

    nanoAOD_tau_switch = cms.PSet(
        idsToAdd = cms.vstring()
    )
    (run2_nanoAOD_106Xv2 | run3_nanoAOD_122).toModify(nanoAOD_tau_switch, idsToAdd = ["deepTau2018v2p5"])
    (run2_nanoAOD_106Xv2 | run3_nanoAOD_122).toModify(process, lambda p : nanoAOD_addTauIds(p, nanoAOD_tau_switch.idsToAdd.value()))
    nanoAOD_boostedTau_switch = cms.PSet(
        idsToAdd = cms.vstring()
    )
    run2_nanoAOD_106Xv2.toModify(nanoAOD_boostedTau_switch, idsToAdd = ["2017v2", "dR0p32017v2", "newDM2017v2","againstEle2018"])
    run2_nanoAOD_106Xv2.toModify(process, lambda p : nanoAOD_addBoostedTauIds(p, nanoAOD_boostedTau_switch.idsToAdd.value()))

    return process

def nanoAOD_customizeData(process):
    process = nanoAOD_customizeCommon(process)
    return process

def nanoAOD_customizeMC(process):
    process = nanoAOD_customizeCommon(process)
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

