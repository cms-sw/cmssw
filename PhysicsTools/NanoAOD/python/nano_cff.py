from __future__ import print_function
import FWCore.ParameterSet.Config as cms
from PhysicsTools.NanoAOD.common_cff import *
from PhysicsTools.NanoAOD.nano_eras_cff import *
from PhysicsTools.NanoAOD.jets_cff import *
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

nanoMetadata = cms.EDProducer("UniqueStringProducer",
    strings = cms.PSet(
        tag = cms.string("untagged"),
    )
)

linkedObjects = cms.EDProducer("PATObjectCrossLinker",
   jets=cms.InputTag("finalJets"),
   muons=cms.InputTag("finalMuons"),
   electrons=cms.InputTag("finalElectrons"),
   taus=cms.InputTag("finalTaus"),
   photons=cms.InputTag("finalPhotons"),
)

simpleCleanerTable = cms.EDProducer("NanoAODSimpleCrossCleaner",
   name=cms.string("cleanmask"),
   doc=cms.string("simple cleaning mask with priority to leptons"),
   jets=cms.InputTag("linkedObjects","jets"),
   muons=cms.InputTag("linkedObjects","muons"),
   electrons=cms.InputTag("linkedObjects","electrons"),
   taus=cms.InputTag("linkedObjects","taus"),
   photons=cms.InputTag("linkedObjects","photons"),
   jetSel=cms.string("pt>15"),
   muonSel=cms.string("track.isNonnull && isLooseMuon && isPFMuon && innerTrack.validFraction >= 0.49 && ( isGlobalMuon && globalTrack.normalizedChi2 < 3 && combinedQuality.chi2LocalPosition < 12 && combinedQuality.trkKink < 20 && segmentCompatibility >= 0.303 || segmentCompatibility >= 0.451 )"),
   electronSel=cms.string(""),
   tauSel=cms.string(""),
   photonSel=cms.string(""),
   jetName=cms.string("Jet"),muonName=cms.string("Muon"),electronName=cms.string("Electron"),
   tauName=cms.string("Tau"),photonName=cms.string("Photon")
)

btagSFdir="PhysicsTools/NanoAOD/data/btagSF/"

btagWeightTable = cms.EDProducer("BTagSFProducer",
    src = cms.InputTag("linkedObjects","jets"),
    cut = cms.string("pt > 25. && abs(eta) < 2.5"),
    discNames = cms.vstring(
        "pfCombinedInclusiveSecondaryVertexV2BJetTags",
        "pfDeepCSVJetTags:probb+pfDeepCSVJetTags:probbb",       #if multiple MiniAOD branches need to be summed up (e.g., DeepCSV b+bb), separate them using '+' delimiter
        "pfCombinedMVAV2BJetTags"
    ),
    discShortNames = cms.vstring(
        "CSVV2",
        "DeepCSVB",
        "CMVA"
    ),
    weightFiles = cms.vstring(                                  #default settings are for 2017 94X. toModify function is called later for other eras.
        btagSFdir+"CSVv2_94XSF_V2_B_F.csv",
        btagSFdir+"DeepCSV_94XSF_V2_B_F.csv",
        "unavailable"                                           #if SFs for an algorithm in an era is unavailable, the corresponding branch will not be stored
    ),
    operatingPoints = cms.vstring("3","3","3"),                 #loose = 0, medium = 1, tight = 2, reshaping = 3
    measurementTypesB = cms.vstring("iterativefit","iterativefit","iterativefit"),     #e.g. "comb", "incl", "ttbar", "iterativefit"
    measurementTypesC = cms.vstring("iterativefit","iterativefit","iterativefit"),
    measurementTypesUDSG = cms.vstring("iterativefit","iterativefit","iterativefit"),
    sysTypes = cms.vstring("central","central","central")
)

for modifier in run2_miniAOD_80XLegacy, run2_nanoAOD_94X2016: # to be updated when SF for Summer16MiniAODv3 MC will be available
    modifier.toModify(btagWeightTable,
        cut = cms.string("pt > 25. && abs(eta) < 2.4"),             #80X corresponds to 2016, |eta| < 2.4
        weightFiles = cms.vstring(                                  #80X corresponds to 2016 SFs
            btagSFdir+"CSVv2_Moriond17_B_H.csv",
            "unavailable",
            btagSFdir+"cMVAv2_Moriond17_B_H.csv"
        )
    )


lhcInfoTable = cms.EDProducer("LHCInfoProducer",
    precision = cms.int32(10),
)


lheInfoTable = cms.EDProducer("LHETablesProducer",
    lheInfo = cms.VInputTag(cms.InputTag("externalLHEProducer"), cms.InputTag("source")),
    precision = cms.int32(14),
    storeLHEParticles = cms.bool(True)
)

l1bits=cms.EDProducer("L1TriggerResultsConverter", src=cms.InputTag("gtStage2Digis"), legacyL1=cms.bool(False),
                      storeUnprefireableBit=cms.bool(True), src_ext=cms.InputTag("gtStage2Digis"))
(run2_miniAOD_80XLegacy | run2_nanoAOD_94X2016 | run2_nanoAOD_94XMiniAODv1 | run2_nanoAOD_94XMiniAODv2 | run2_nanoAOD_102Xv1).toModify(l1bits, storeUnprefireableBit=False)

nanoSequenceCommon = cms.Sequence(
        nanoMetadata + jetSequence + muonSequence + tauSequence + boostedTauSequence + electronSequence + lowPtElectronSequence + photonSequence+vertexSequence+
        isoTrackSequence + jetLepSequence + # must be after all the leptons
        linkedObjects  +
        jetTables + muonTables + tauTables + boostedTauTables + electronTables + lowPtElectronTables + photonTables +  globalTables +vertexTables+ metTables+simpleCleanerTable + isoTrackTables
        )
#remove boosted tau from previous eras
(run2_miniAOD_80XLegacy | run2_nanoAOD_92X | run2_nanoAOD_94XMiniAODv1 | run2_nanoAOD_94X2016 | run2_nanoAOD_94XMiniAODv2 | run2_nanoAOD_102Xv1 | run2_nanoAOD_106Xv1).toReplaceWith(nanoSequenceCommon, nanoSequenceCommon.copyAndExclude([boostedTauSequence, boostedTauTables]))

nanoSequenceOnlyFullSim = cms.Sequence(triggerObjectTables + l1bits)

nanoSequenceOnlyData = cms.Sequence(protonTables + lhcInfoTable)

nanoSequence = cms.Sequence(nanoSequenceCommon + nanoSequenceOnlyData + nanoSequenceOnlyFullSim)

( run2_nanoAOD_106Xv1 & ~run2_nanoAOD_devel).toReplaceWith(nanoSequence, nanoSequence.copyAndExclude([nanoSequenceOnlyData]))

nanoSequenceFS = cms.Sequence(genParticleSequence + genVertexTables + particleLevelSequence + nanoSequenceCommon + jetMC + muonMC + electronMC + lowPtElectronMC + photonMC + tauMC + boostedTauMC + metMC + ttbarCatMCProducers +  globalTablesMC + btagWeightTable + genWeightsTable + genVertexTable + genParticleTables + genProtonTables + particleLevelTables + lheInfoTable  + ttbarCategoryTable )

(run2_nanoAOD_92X | run2_miniAOD_80XLegacy | run2_nanoAOD_94X2016 | run2_nanoAOD_94X2016 | \
    run2_nanoAOD_94XMiniAODv1 | run2_nanoAOD_94XMiniAODv2 | \
    run2_nanoAOD_102Xv1 ).toReplaceWith(nanoSequenceFS, nanoSequenceFS.copyAndExclude([genVertexTable, genVertexT0Table]))

#remove boosted tau from previous eras
(run2_miniAOD_80XLegacy | run2_nanoAOD_92X | run2_nanoAOD_94XMiniAODv1 | run2_nanoAOD_94X2016 | run2_nanoAOD_94XMiniAODv2 | run2_nanoAOD_102Xv1 | run2_nanoAOD_106Xv1).toReplaceWith(nanoSequenceFS, nanoSequenceFS.copyAndExclude([boostedTauMC]))

# GenVertex only stored in newer MiniAOD
nanoSequenceMC = nanoSequenceFS.copy()
nanoSequenceMC.insert(nanoSequenceFS.index(nanoSequenceCommon)+1,nanoSequenceOnlyFullSim)

# modify extraFlagsTable to store ecalBadCalibFilter decision which is re-run with updated bad crystal list for 2017 and 2018 samples
for modifier in run2_nanoAOD_94XMiniAODv1, run2_nanoAOD_94XMiniAODv2, run2_nanoAOD_102Xv1:
    modifier.toModify(extraFlagsTable, variables= cms.PSet())
    modifier.toModify(extraFlagsTable, variables = dict(Flag_ecalBadCalibFilterV2 = ExtVar(cms.InputTag("ecalBadCalibFilterNanoTagger"), bool, doc = "Bad ECAL calib flag (updated xtal list)")))

# modifier which adds new tauIDs (currently only deepTauId2017v2p1 is being added)
import RecoTauTag.RecoTau.tools.runTauIdMVA as tauIdConfig
def nanoAOD_addTauIds(process):
    updatedTauName = "slimmedTausUpdated"
    tauIdEmbedder = tauIdConfig.TauIDEmbedder(process, cms, debug = False, updatedTauName = updatedTauName,
            toKeep = [ "deepTau2017v2p1" ])
    tauIdEmbedder.runTauID()
    process.patTauMVAIDsSeq.insert(process.patTauMVAIDsSeq.index(getattr(process, updatedTauName)),
                                   process.rerunMvaIsolationSequence)
    return process

def nanoAOD_addBoostedTauIds(process):
    updatedBoostedTauName = "slimmedTausBoostedNewID"
    boostedTauIdEmbedder = tauIdConfig.TauIDEmbedder(process, cms, debug=False, 
                                                     originalTauName = "slimmedTausBoosted",
                                                     updatedTauName = updatedBoostedTauName,
                                                     postfix="Boosted",
                                                     toKeep = [ "2017v2", "dR0p32017v2", "newDM2017v2", "againstEle2018",])
    boostedTauIdEmbedder.runTauID()
    process.boostedTauSequence.insert(process.boostedTauSequence.index(process.finalBoostedTaus),
                                      process.rerunMvaIsolationSequenceBoosted)

    process.boostedTauSequence.insert(process.boostedTauSequence.index(process.finalBoostedTaus),
                                      getattr(process, updatedBoostedTauName))

    return process
 

from PhysicsTools.PatAlgos.tools.jetTools import updateJetCollection
def nanoAOD_addDeepInfo(process,addDeepBTag,addDeepFlavour):
    _btagDiscriminators=[]
    if addDeepBTag:
        print("Updating process to run DeepCSV btag")
        _btagDiscriminators += ['pfDeepCSVJetTags:probb','pfDeepCSVJetTags:probbb','pfDeepCSVJetTags:probc']
    if addDeepFlavour:
        print("Updating process to run DeepFlavour btag")
        _btagDiscriminators += ['pfDeepFlavourJetTags:probb','pfDeepFlavourJetTags:probbb','pfDeepFlavourJetTags:problepb','pfDeepFlavourJetTags:probc']
    if len(_btagDiscriminators)==0: return process
    print("Will recalculate the following discriminators: "+", ".join(_btagDiscriminators))
    updateJetCollection(
               process,
               jetSource = cms.InputTag('slimmedJets'),
               jetCorrections = ('AK4PFchs', cms.vstring(['L1FastJet', 'L2Relative', 'L3Absolute','L2L3Residual']), 'None'),
               btagDiscriminators = _btagDiscriminators,
               postfix = 'WithDeepInfo',
           )
    process.load("Configuration.StandardSequences.MagneticField_cff")
    process.jetCorrFactorsNano.src="selectedUpdatedPatJetsWithDeepInfo"
    process.updatedJets.jetSource="selectedUpdatedPatJetsWithDeepInfo"
    return process

def nanoAOD_addDeepMET(process, addDeepMETProducer, ResponseTune_Graph):
    if addDeepMETProducer:
        # produce DeepMET on the fly if it is not in MiniAOD
        print("add DeepMET Producers")
        process.load('RecoMET.METPUSubtraction.deepMETProducer_cfi')
        process.deepMETsResolutionTune = process.deepMETProducer.clone()
        process.deepMETsResponseTune = process.deepMETProducer.clone()
        #process.deepMETsResponseTune.graph_path = 'RecoMET/METPUSubtraction/data/deepmet/deepmet_resp_v1_2018.pb'
        process.deepMETsResponseTune.graph_path = ResponseTune_Graph.value()
    process.metTables += process.deepMetTables
    return process

from PhysicsTools.PatUtils.tools.runMETCorrectionsAndUncertainties import runMetCorAndUncFromMiniAOD
from PhysicsTools.PatAlgos.slimming.puppiForMET_cff import makePuppiesFromMiniAOD
def nanoAOD_recalibrateMETs(process,isData):
    # add DeepMETs
    nanoAOD_DeepMET_switch = cms.PSet(
        nanoAOD_addDeepMET_switch = cms.untracked.bool(True), # decide if DeeMET should be included in Nano
        nanoAOD_produceDeepMET_switch = cms.untracked.bool(False), # decide if DeepMET should be computed on the fly
        ResponseTune_Graph = cms.untracked.string('RecoMET/METPUSubtraction/data/deepmet/deepmet_resp_v1_2018.pb')
    )
    # compute DeepMETs for the eras before UL-ReminiAOD
    (~(run2_nanoAOD_106Xv2 | run2_miniAOD_devel)).toModify(nanoAOD_DeepMET_switch, nanoAOD_produceDeepMET_switch =  cms.untracked.bool(True))
    for modifier in run2_miniAOD_80XLegacy, run2_nanoAOD_94X2016:
        modifier.toModify(nanoAOD_DeepMET_switch, ResponseTune_Graph=cms.untracked.string("RecoMET/METPUSubtraction/data/deepmet/deepmet_resp_v1_2016.pb"))
    if nanoAOD_DeepMET_switch.nanoAOD_addDeepMET_switch:
        process = nanoAOD_addDeepMET(process,
                                     addDeepMETProducer=nanoAOD_DeepMET_switch.nanoAOD_produceDeepMET_switch,
                                     ResponseTune_Graph=nanoAOD_DeepMET_switch.ResponseTune_Graph)

    # if included in Nano, and not computed in the fly, then it should be extracted from minAOD
    extractDeepMETs = nanoAOD_DeepMET_switch.nanoAOD_addDeepMET_switch and not nanoAOD_DeepMET_switch.nanoAOD_produceDeepMET_switch

    runMetCorAndUncFromMiniAOD(process,isData=isData, extractDeepMETs=extractDeepMETs)
    process.nanoSequenceCommon.insert(process.nanoSequenceCommon.index(process.jetSequence),cms.Sequence(process.fullPatMetSequence))
    process.basicJetsForMetForT1METNano = process.basicJetsForMet.clone(
        src = process.updatedJetsWithUserData.src,
        skipEM = False,
        type1JetPtThreshold = 0.0,
        calcMuonSubtrRawPtAsValueMap = cms.bool(True),
    )
    process.jetSequence.insert(process.jetSequence.index(process.updatedJetsWithUserData),cms.Sequence(process.basicJetsForMetForT1METNano))
    process.updatedJetsWithUserData.userFloats.muonSubtrRawPt = cms.InputTag("basicJetsForMetForT1METNano:MuonSubtrRawPt")
    process.corrT1METJetTable.src = process.finalJets.src
    process.corrT1METJetTable.cut = "pt<15 && abs(eta)<9.9"
    for table in process.jetTable, process.corrT1METJetTable:
        table.variables.muonSubtrFactor = Var("1-userFloat('muonSubtrRawPt')/(pt()*jecFactor('Uncorrected'))",float,doc="1-(muon-subtracted raw pt)/(raw pt)",precision=6)
    process.metTables += process.corrT1METJetTable
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
    
    runMetCorAndUncFromMiniAOD(process,isData=isData,metType="Puppi",postfix="Puppi",jetFlavor="AK4PFPuppi", recoMetFromPFCs=bool(nanoAOD_PuppiV15_switch.recoMetFromPFCs), reclusterJets=bool(nanoAOD_PuppiV15_switch.reclusterJets))
    process.nanoSequenceCommon.insert(process.nanoSequenceCommon.index(process.jetSequence),cms.Sequence(process.puppiMETSequence+process.fullPatMetSequencePuppi))
    return process

from PhysicsTools.SelectorUtils.tools.vid_id_tools import *
def nanoAOD_activateVID(process):
    switchOnVIDElectronIdProducer(process,DataFormat.MiniAOD)
    for modname in electron_id_modules_WorkingPoints_nanoAOD.modules:
        setupAllVIDIdsInModule(process,modname,setupVIDElectronSelection)
    process.electronSequence.insert(process.electronSequence.index(process.bitmapVIDForEle),process.egmGsfElectronIDSequence)
    for modifier in run2_miniAOD_80XLegacy,run2_nanoAOD_94XMiniAODv1,run2_nanoAOD_94XMiniAODv2,run2_nanoAOD_94X2016,run2_nanoAOD_102Xv1,run2_nanoAOD_106Xv1:
        modifier.toModify(process.electronMVAValueMapProducer, srcMiniAOD = "slimmedElectronsUpdated")
        modifier.toModify(process.egmGsfElectronIDs, physicsObjectSrc = "slimmedElectronsUpdated")

    run2_nanoAOD_106Xv2.toModify(process.electronMVAValueMapProducer, src = "slimmedElectrons")
    run2_nanoAOD_106Xv2.toModify(process.egmGsfElectronIDs, physicsObjectSrc = "slimmedElectrons")
    switchOnVIDPhotonIdProducer(process,DataFormat.MiniAOD) # do not call this to avoid resetting photon IDs in VID, if called before inside makePuppiesFromMiniAOD
    for modname in photon_id_modules_WorkingPoints_nanoAOD.modules:
        setupAllVIDIdsInModule(process,modname,setupVIDPhotonSelection)
    process.photonSequence.insert(process.photonSequence.index(bitmapVIDForPho),process.egmPhotonIDSequence)
    for modifier in run2_miniAOD_80XLegacy,run2_nanoAOD_94XMiniAODv1,run2_nanoAOD_94XMiniAODv2,run2_nanoAOD_94X2016 ,run2_nanoAOD_102Xv1:
        modifier.toModify(process.photonMVAValueMapProducer, srcMiniAOD = "slimmedPhotonsTo106X")
        modifier.toModify(process.egmPhotonIDs, physicsObjectSrc = "slimmedPhotonsTo106X")
    return process

def nanoAOD_addDeepInfoAK8(process, addDeepBTag, addDeepBoostedJet, addDeepDoubleX, addDeepDoubleXV2, addParticleNet, addParticleNetMass, jecPayload):
    _btagDiscriminators=[]
    if addDeepBTag:
        print("Updating process to run DeepCSV btag to AK8 jets")
        _btagDiscriminators += ['pfDeepCSVJetTags:probb','pfDeepCSVJetTags:probbb']
    if addDeepBoostedJet:
        print("Updating process to run DeepBoostedJet on datasets before 103X")
        from RecoBTag.ONNXRuntime.pfDeepBoostedJet_cff import _pfDeepBoostedJetTagsAll as pfDeepBoostedJetTagsAll
        _btagDiscriminators += pfDeepBoostedJetTagsAll
    if addParticleNet:
        print("Updating process to run ParticleNet before it's included in MiniAOD")
        from RecoBTag.ONNXRuntime.pfParticleNet_cff import _pfParticleNetJetTagsAll as pfParticleNetJetTagsAll
        _btagDiscriminators += pfParticleNetJetTagsAll
    if addParticleNetMass:
        from RecoBTag.ONNXRuntime.pfParticleNet_cff import _pfParticleNetMassRegressionOutputs
        _btagDiscriminators += _pfParticleNetMassRegressionOutputs
    if addDeepDoubleX:
        print("Updating process to run DeepDoubleX on datasets before 104X")
        _btagDiscriminators += ['pfDeepDoubleBvLJetTags:probHbb', \
            'pfDeepDoubleCvLJetTags:probHcc', \
            'pfDeepDoubleCvBJetTags:probHcc', \
            'pfMassIndependentDeepDoubleBvLJetTags:probHbb', 'pfMassIndependentDeepDoubleCvLJetTags:probHcc', 'pfMassIndependentDeepDoubleCvBJetTags:probHcc']
    if addDeepDoubleXV2:
        print("Updating process to run DeepDoubleXv2 on datasets before 11X")
        _btagDiscriminators += [
            'pfMassIndependentDeepDoubleBvLV2JetTags:probHbb',
            'pfMassIndependentDeepDoubleCvLV2JetTags:probHcc',
            'pfMassIndependentDeepDoubleCvBV2JetTags:probHcc'
            ]
    if len(_btagDiscriminators)==0: return process
    print("Will recalculate the following discriminators on AK8 jets: "+", ".join(_btagDiscriminators))
    updateJetCollection(
       process,
       jetSource = cms.InputTag('slimmedJetsAK8'),
       pvSource = cms.InputTag('offlineSlimmedPrimaryVertices'),
       svSource = cms.InputTag('slimmedSecondaryVertices'),
       rParam = 0.8,
       jetCorrections = (jecPayload.value(), cms.vstring(['L1FastJet', 'L2Relative', 'L3Absolute', 'L2L3Residual']), 'None'),
       btagDiscriminators = _btagDiscriminators,
       postfix='AK8WithDeepInfo',
       printWarning = False
       )
    process.jetCorrFactorsAK8.src="selectedUpdatedPatJetsAK8WithDeepInfo"
    process.updatedJetsAK8.jetSource="selectedUpdatedPatJetsAK8WithDeepInfo"
    return process

from PhysicsTools.PatUtils.tools.runMETCorrectionsAndUncertainties import runMetCorAndUncFromMiniAOD
def nanoAOD_runMETfixEE2017(process,isData):
    runMetCorAndUncFromMiniAOD(process,isData=isData,
                               fixEE2017 = True,
                               fixEE2017Params = {'userawPt': True, 'ptThreshold':50.0, 'minEtaThreshold':2.65, 'maxEtaThreshold': 3.139},
                               postfix = "FixEE2017")
    process.nanoSequenceCommon.insert(process.nanoSequenceCommon.index(jetSequence),process.fullPatMetSequenceFixEE2017)

def nanoAOD_customizeCommon(process):
    makePuppiesFromMiniAOD(process,True)
    process.puppiNoLep.useExistingWeights = True
    process.puppi.useExistingWeights = True
    run2_nanoAOD_106Xv1.toModify(process.puppiNoLep, useExistingWeights = False)
    run2_nanoAOD_106Xv1.toModify(process.puppi, useExistingWeights = False)
    process = nanoAOD_activateVID(process)
    nanoAOD_addDeepInfo_switch = cms.PSet(
        nanoAOD_addDeepBTag_switch = cms.untracked.bool(False),
        nanoAOD_addDeepFlavourTag_switch = cms.untracked.bool(False),
        )
    run2_miniAOD_80XLegacy.toModify(nanoAOD_addDeepInfo_switch, nanoAOD_addDeepBTag_switch = cms.untracked.bool(True))
    for modifier in run2_miniAOD_80XLegacy, run2_nanoAOD_94X2016, run2_nanoAOD_94XMiniAODv1, run2_nanoAOD_94XMiniAODv2:
        modifier.toModify(nanoAOD_addDeepInfo_switch, nanoAOD_addDeepFlavourTag_switch =  cms.untracked.bool(True))
    process = nanoAOD_addDeepInfo(process,
                                  addDeepBTag=nanoAOD_addDeepInfo_switch.nanoAOD_addDeepBTag_switch,
                                  addDeepFlavour=nanoAOD_addDeepInfo_switch.nanoAOD_addDeepFlavourTag_switch)
    nanoAOD_addDeepInfoAK8_switch = cms.PSet(
        nanoAOD_addDeepBTag_switch = cms.untracked.bool(False),
        nanoAOD_addDeepBoostedJet_switch = cms.untracked.bool(True),
        nanoAOD_addDeepDoubleX_switch = cms.untracked.bool(True),
        nanoAOD_addDeepDoubleXV2_switch = cms.untracked.bool(True),
        nanoAOD_addParticleNet_switch = cms.untracked.bool(True),
        nanoAOD_addParticleNetMass_switch = cms.untracked.bool(True),
        jecPayload = cms.untracked.string('AK8PFPuppi')
        )
    # Don't run on old mini due to compatibility
    # 80X contains ak8PFJetsCHS jets instead of puppi
    run2_miniAOD_80XLegacy.toModify(nanoAOD_addDeepInfoAK8_switch,
                                    nanoAOD_addDeepBTag_switch = True,
                                    nanoAOD_addDeepBoostedJet_switch = False,
                                    nanoAOD_addDeepDoubleX_switch = False,
                                    nanoAOD_addDeepDoubleXV2_switch = False,
                                    nanoAOD_addParticleNet_switch = False,
                                    nanoAOD_addParticleNetMass_switch = False,
                                    jecPayload = 'AK8PFchs')
    # Don't rerun where already present
    (run2_miniAOD_devel).toModify(
        nanoAOD_addDeepInfoAK8_switch,
        nanoAOD_addParticleNetMass_switch = False,
        )
    (run2_nanoAOD_106Xv2 | run2_miniAOD_devel).toModify(
        nanoAOD_addDeepInfoAK8_switch,
        nanoAOD_addDeepBoostedJet_switch = False,
        nanoAOD_addDeepDoubleX_switch = False,
        nanoAOD_addDeepDoubleXV2_switch = False,
        nanoAOD_addParticleNet_switch = False,
        )
    run2_nanoAOD_106Xv1.toModify(
         nanoAOD_addDeepInfoAK8_switch,
         nanoAOD_addDeepBoostedJet_switch = False,
         nanoAOD_addDeepDoubleX_switch = False,
         )
    # no-change policy
    (run2_nanoAOD_106Xv1 & ~run2_nanoAOD_devel).toModify(
        nanoAOD_addDeepInfoAK8_switch,
        nanoAOD_addParticleNetMass_switch = False,
        )
    process = nanoAOD_addDeepInfoAK8(process,
                                     addDeepBTag=nanoAOD_addDeepInfoAK8_switch.nanoAOD_addDeepBTag_switch,
                                     addDeepBoostedJet=nanoAOD_addDeepInfoAK8_switch.nanoAOD_addDeepBoostedJet_switch,
                                     addDeepDoubleX=nanoAOD_addDeepInfoAK8_switch.nanoAOD_addDeepDoubleX_switch,
                                     addDeepDoubleXV2=nanoAOD_addDeepInfoAK8_switch.nanoAOD_addDeepDoubleXV2_switch,
                                     addParticleNet=nanoAOD_addDeepInfoAK8_switch.nanoAOD_addParticleNet_switch,
                                     addParticleNetMass=nanoAOD_addDeepInfoAK8_switch.nanoAOD_addParticleNetMass_switch,
                                     jecPayload=nanoAOD_addDeepInfoAK8_switch.jecPayload)
    addTauIds_switch = cms.PSet(
        nanoAOD_addTauIds_switch = cms.untracked.bool(True),
        nanoAOD_addBoostedTauIds_switch = cms.untracked.bool(True)
    )
    ((run2_nanoAOD_106Xv2 | run2_miniAOD_devel | run2_tau_ul_2016 | run2_tau_ul_2018) & \
    (~(run2_nanoAOD_94X2016 | run2_nanoAOD_94XMiniAODv1 | run2_nanoAOD_94XMiniAODv2 | run2_nanoAOD_102Xv1 | run2_nanoAOD_106Xv1))).toModify(addTauIds_switch, nanoAOD_addTauIds_switch = False)
    (run2_miniAOD_80XLegacy | run2_nanoAOD_92X | run2_nanoAOD_94X2016 | run2_nanoAOD_94XMiniAODv1 | run2_nanoAOD_94XMiniAODv2 | run2_nanoAOD_102Xv1 | run2_nanoAOD_106Xv1).toModify(addTauIds_switch, nanoAOD_addBoostedTauIds_switch = False)
    if addTauIds_switch.nanoAOD_addTauIds_switch:
        process = nanoAOD_addTauIds(process)
    if addTauIds_switch.nanoAOD_addBoostedTauIds_switch:
        process = nanoAOD_addBoostedTauIds(process)
    return process

def nanoAOD_customizeData(process):
    process = nanoAOD_customizeCommon(process)
    process = nanoAOD_recalibrateMETs(process,isData=True)
    for modifier in run2_nanoAOD_94XMiniAODv1, run2_nanoAOD_94XMiniAODv2:
        modifier.toModify(process, lambda p: nanoAOD_runMETfixEE2017(p,isData=True))
    return process

def nanoAOD_customizeMC(process):
    process = nanoAOD_customizeCommon(process)
    process = nanoAOD_recalibrateMETs(process,isData=False)
    for modifier in run2_nanoAOD_94XMiniAODv1, run2_nanoAOD_94XMiniAODv2:
        modifier.toModify(process, lambda p: nanoAOD_runMETfixEE2017(p,isData=False))
    return process

def nanoWmassGenCustomize(process):
    pdgSelection="?(abs(pdgId) == 11|| abs(pdgId)==13 || abs(pdgId)==15 ||abs(pdgId)== 12 || abs(pdgId)== 14 || abs(pdgId)== 16|| abs(pdgId)==24|| pdgId== 23)"
    # Keep precision same as default RECO for selected particles                                                                                                                                                                                                                  
    ptPrecision="{}?{}:{}".format(pdgSelection, CandVars.pt.precision.value(),genParticleTable.variables.pt.precision.value())
    process.genParticleTable.variables.pt.precision=cms.string(ptPrecision)
    phiPrecision="{} ? {} : {}".format(pdgSelection, CandVars.phi.precision.value(), genParticleTable.variables.phi.precision.value())
    process.genParticleTable.variables.phi.precision=cms.string(phiPrecision)
    etaPrecision="{} ? {} : {}".format(pdgSelection, CandVars.eta.precision.value(), genParticleTable.variables.eta.precision.value())
    process.genParticleTable.variables.eta.precision=cms.string(etaPrecision)
    return process

### Era dependent customization
_80x_sequence = nanoSequenceCommon.copy()
#remove stuff
_80x_sequence.remove(isoTrackTables)
_80x_sequence.remove(isoTrackSequence)
#add stuff
_80x_sequence.insert(_80x_sequence.index(jetSequence), extraFlagsProducers)
_80x_sequence.insert(_80x_sequence.index(simpleCleanerTable)+1, extraFlagsTable)

run2_miniAOD_80XLegacy.toReplaceWith( nanoSequenceCommon, _80x_sequence)

_102x_sequence = nanoSequenceCommon.copy()
#add stuff
_102x_sequence.insert(_102x_sequence.index(jetSequence),extraFlagsProducers102x)
_102x_sequence.insert(_102x_sequence.index(simpleCleanerTable)+1,extraFlagsTable)

for modifier in run2_nanoAOD_94XMiniAODv1, run2_nanoAOD_94XMiniAODv2, run2_nanoAOD_102Xv1:
    modifier.toReplaceWith(nanoSequenceCommon, _102x_sequence)
