from __future__ import print_function
import FWCore.ParameterSet.Config as cms
from PhysicsTools.NanoAOD.common_cff import *
from PhysicsTools.NanoAOD.jets_cff import *
from PhysicsTools.NanoAOD.muons_cff import *
from PhysicsTools.NanoAOD.taus_cff import *
from PhysicsTools.NanoAOD.electrons_cff import *
from PhysicsTools.NanoAOD.photons_cff import *
from PhysicsTools.NanoAOD.globals_cff import *
from PhysicsTools.NanoAOD.extraflags_cff import *
from PhysicsTools.NanoAOD.ttbarCategorization_cff import *
from PhysicsTools.NanoAOD.genparticles_cff import *
from PhysicsTools.NanoAOD.particlelevel_cff import *
from PhysicsTools.NanoAOD.vertices_cff import *
from PhysicsTools.NanoAOD.met_cff import *
from PhysicsTools.NanoAOD.triggerObjects_cff import *
from PhysicsTools.NanoAOD.isotracks_cff import *
from PhysicsTools.NanoAOD.NanoAODEDMEventContent_cff import *

from Configuration.Eras.Modifier_run2_miniAOD_80XLegacy_cff import run2_miniAOD_80XLegacy
from Configuration.Eras.Modifier_run2_nanoAOD_92X_cff import run2_nanoAOD_92X
from Configuration.Eras.Modifier_run2_nanoAOD_94X2016_cff import run2_nanoAOD_94X2016
from Configuration.Eras.Modifier_run2_nanoAOD_94XMiniAODv1_cff import run2_nanoAOD_94XMiniAODv1
from Configuration.Eras.Modifier_run2_nanoAOD_94XMiniAODv2_cff import run2_nanoAOD_94XMiniAODv2

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
   muonSel=cms.string("isPFMuon && innerTrack.validFraction >= 0.49 && ( isGlobalMuon && globalTrack.normalizedChi2 < 3 && combinedQuality.chi2LocalPosition < 12 && combinedQuality.trkKink < 20 && segmentCompatibility >= 0.303 || segmentCompatibility >= 0.451 )"),
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

run2_nanoAOD_92X.toModify(btagWeightTable,                      #92X corresponds to MCv1, for which SFs are unavailable
    weightFiles = cms.vstring(
        "unavailable",
        "unavailable",                    
        "unavailable"                                            
    )
)                    

genWeightsTable = cms.EDProducer("GenWeightsTableProducer",
    genEvent = cms.InputTag("generator"),
    lheInfo = cms.InputTag("externalLHEProducer"),
    preferredPDFs = cms.VPSet( # see https://lhapdf.hepforge.org/pdfsets.html
        cms.PSet( name = cms.string("PDF4LHC15_nnlo_30_pdfas"), lhaid = cms.uint32(91400) ),
        cms.PSet( name = cms.string("NNPDF31_nnlo_hessian_pdfas"), lhaid = cms.uint32(306000) ),
        cms.PSet( name = cms.string("NNPDF30_nlo_as_0118"), lhaid = cms.uint32(260000) ), # for some 92X samples. Note that the nominal weight, 260000, is not included in the LHE ...
        cms.PSet( name = cms.string("NNPDF30_lo_as_0130"), lhaid = cms.uint32(262000) ), # some MLM 80X samples have only this (e.g. /store/mc/RunIISummer16MiniAODv2/DYJetsToLL_M-50_TuneCUETP8M1_13TeV-madgraphMLM-pythia8/MINIAODSIM/PUMoriond17_80X_mcRun2_asymptotic_2016_TrancheIV_v6_ext1-v2/120000/02A210D6-F5C3-E611-B570-008CFA197BD4.root )
        cms.PSet( name = cms.string("NNPDF30_nlo_nf_4_pdfas"), lhaid = cms.uint32(292000) ), # some FXFX 80X samples have only this (e.g. WWTo1L1Nu2Q, WWTo4Q)
        cms.PSet( name = cms.string("NNPDF30_nlo_nf_5_pdfas"), lhaid = cms.uint32(292200) ), # some FXFX 80X samples have only this (e.g. DYJetsToLL_Pt, WJetsToLNu_Pt, DYJetsToNuNu_Pt)
    ),
    namedWeightIDs = cms.vstring(),
    namedWeightLabels = cms.vstring(),
    lheWeightPrecision = cms.int32(14),
    maxPdfWeights = cms.uint32(150), 
    debug = cms.untracked.bool(False),
)
lheInfoTable = cms.EDProducer("LHETablesProducer",
    lheInfo = cms.InputTag("externalLHEProducer"),
    precision = cms.int32(14),
    storeLHEParticles = cms.bool(True) 
)

l1bits=cms.EDProducer("L1TriggerResultsConverter", src=cms.InputTag("gtStage2Digis"), legacyL1=cms.bool(False))

nanoSequenceCommon = cms.Sequence(
        nanoMetadata + jetSequence + muonSequence + tauSequence + electronSequence+photonSequence+vertexSequence+
        isoTrackSequence + # must be after all the leptons 
        linkedObjects  +
        jetTables + muonTables + tauTables + electronTables + photonTables +  globalTables +vertexTables+ metTables+simpleCleanerTable + isoTrackTables
        )
nanoSequenceOnlyFullSim = cms.Sequence(triggerObjectTables + l1bits)

nanoSequence = cms.Sequence(nanoSequenceCommon + nanoSequenceOnlyFullSim)

nanoSequenceFS = cms.Sequence(genParticleSequence + particleLevelSequence + nanoSequenceCommon + jetMC + muonMC + electronMC + photonMC + tauMC + metMC + ttbarCatMCProducers +  globalTablesMC + btagWeightTable + genWeightsTable + genParticleTables + particleLevelTables + lheInfoTable  + ttbarCategoryTable )

nanoSequenceMC = nanoSequenceFS.copy()
nanoSequenceMC.insert(nanoSequenceFS.index(nanoSequenceCommon)+1,nanoSequenceOnlyFullSim)


from PhysicsTools.PatAlgos.tools.jetTools import updateJetCollection
def nanoAOD_addDeepInfo(process,addDeepBTag,addDeepFlavour):
    _btagDiscriminators=[]
    if addDeepBTag:
        print("Updating process to run DeepCSV btag")
        _btagDiscriminators += ['pfDeepCSVJetTags:probb','pfDeepCSVJetTags:probbb','pfDeepCSVJetTags:probc']
    if addDeepFlavour:
        print("Updating process to run DeepFlavour btag")
        _btagDiscriminators += ['pfDeepFlavourJetTags:probb','pfDeepFlavourJetTags:probbb','pfDeepFlavourJetTags:problepb']
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
    process.looseJetId.src="selectedUpdatedPatJetsWithDeepInfo"
    process.tightJetId.src="selectedUpdatedPatJetsWithDeepInfo"
    process.tightJetIdLepVeto.src="selectedUpdatedPatJetsWithDeepInfo"
    process.bJetVars.src="selectedUpdatedPatJetsWithDeepInfo"
    process.slimmedJetsWithUserData.src="selectedUpdatedPatJetsWithDeepInfo"
    process.qgtagger80x.srcJets="selectedUpdatedPatJetsWithDeepInfo"
    if addDeepFlavour:
        process.pfDeepFlavourJetTagsWithDeepInfo.graph_path = 'RecoBTag/Combined/data/DeepFlavourV03_10X_training/constant_graph.pb'
        process.pfDeepFlavourJetTagsWithDeepInfo.lp_names = ["cpf_input_batchnorm/keras_learning_phase"]
    return process

from PhysicsTools.PatUtils.tools.runMETCorrectionsAndUncertainties import runMetCorAndUncFromMiniAOD
#from PhysicsTools.PatAlgos.slimming.puppiForMET_cff import makePuppiesFromMiniAOD
def nanoAOD_recalibrateMETs(process,isData):
    runMetCorAndUncFromMiniAOD(process,isData=isData)
    process.nanoSequenceCommon.insert(process.nanoSequenceCommon.index(jetSequence),cms.Sequence(process.fullPatMetSequence))
#    makePuppiesFromMiniAOD(process,True) # call this before in the global customizer otherwise it would reset photon IDs in VID
#    runMetCorAndUncFromMiniAOD(process,isData=isData,metType="Puppi",postfix="Puppi",jetFlavor="AK4PFPuppi")
#    process.puppiNoLep.useExistingWeights = False
#    process.puppi.useExistingWeights = False
#    process.nanoSequenceCommon.insert(process.nanoSequenceCommon.index(jetSequence),cms.Sequence(process.puppiMETSequence+process.fullPatMetSequencePuppi))
    return process

from PhysicsTools.SelectorUtils.tools.vid_id_tools import *
def nanoAOD_activateVID(process):
    switchOnVIDElectronIdProducer(process,DataFormat.MiniAOD)
    for modname in electron_id_modules_WorkingPoints_nanoAOD.modules:
        setupAllVIDIdsInModule(process,modname,setupVIDElectronSelection)
    process.electronSequence.insert(process.electronSequence.index(bitmapVIDForEle),process.egmGsfElectronIDSequence)
    for modifier in run2_miniAOD_80XLegacy, run2_nanoAOD_92X:
        modifier.toModify(process.electronMVAValueMapProducer, srcMiniAOD = "slimmedElectronsUpdated")
        modifier.toModify(process.egmGsfElectronIDs, physicsObjectSrc = "slimmedElectronsUpdated")
        if hasattr(process,"heepIDVarValueMaps"):
            modifier.toModify(process.heepIDVarValueMaps, elesMiniAOD = "slimmedElectronsUpdated")
    switchOnVIDPhotonIdProducer(process,DataFormat.MiniAOD) # do not call this to avoid resetting photon IDs in VID, if called before inside makePuppiesFromMiniAOD
    for modname in photon_id_modules_WorkingPoints_nanoAOD.modules:
        setupAllVIDIdsInModule(process,modname,setupVIDPhotonSelection)
    process.photonSequence.insert(process.photonSequence.index(bitmapVIDForPho),process.egmPhotonIDSequence)
    return process

def nanoAOD_addDeepInfoAK8(process,addDeepBTag,addDeepBoostedJet,jecPayload):
    _btagDiscriminators=[]
    if addDeepBTag:
        print("Updating process to run DeepCSV btag to AK8 jets")
        _btagDiscriminators += ['pfDeepCSVJetTags:probb','pfDeepCSVJetTags:probbb']
    if addDeepBoostedJet:
        print("Updating process to run DeepBoostedJet on datasets before 103X")
        from RecoBTag.MXNet.pfDeepBoostedJet_cff import _pfDeepBoostedJetTagsAll as pfDeepBoostedJetTagsAll
        _btagDiscriminators += pfDeepBoostedJetTagsAll
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
    process.looseJetIdAK8.src = "selectedUpdatedPatJetsAK8WithDeepInfo"
    process.tightJetIdAK8.src = "selectedUpdatedPatJetsAK8WithDeepInfo"
    process.tightJetIdLepVetoAK8.src = "selectedUpdatedPatJetsAK8WithDeepInfo"
    process.slimmedJetsAK8WithUserData.src = "selectedUpdatedPatJetsAK8WithDeepInfo"
    return process

from PhysicsTools.PatUtils.tools.runMETCorrectionsAndUncertainties import runMetCorAndUncFromMiniAOD
def nanoAOD_runMETfixEE2017(process,isData):
    runMetCorAndUncFromMiniAOD(process,isData=isData,
                               fixEE2017 = True,
                               fixEE2017Params = {'userawPt': True, 'ptThreshold':50.0, 'minEtaThreshold':2.65, 'maxEtaThreshold': 3.139},
                               postfix = "FixEE2017")
    process.nanoSequenceCommon.insert(process.nanoSequenceCommon.index(jetSequence),process.fullPatMetSequenceFixEE2017)

def nanoAOD_customizeCommon(process):
#    makePuppiesFromMiniAOD(process,True) # call this here as it calls switchOnVIDPhotonIdProducer
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
        nanoAOD_addDeepBoostedJet_switch = cms.untracked.bool(True), # will deactivate this in future miniAOD releases
        jecPayload = cms.untracked.string('AK8PFPuppi')
        )
    # deepAK8 should not run on 80X, that contains ak8PFJetsCHS jets
    run2_miniAOD_80XLegacy.toModify(nanoAOD_addDeepInfoAK8_switch,
                                    nanoAOD_addDeepBTag_switch = cms.untracked.bool(True),
                                    nanoAOD_addDeepBoostedJet_switch = cms.untracked.bool(False),
                                    jecPayload = cms.untracked.string('AK8PFchs'))
    process = nanoAOD_addDeepInfoAK8(process,
                                     addDeepBTag=nanoAOD_addDeepInfoAK8_switch.nanoAOD_addDeepBTag_switch,
                                     addDeepBoostedJet=nanoAOD_addDeepInfoAK8_switch.nanoAOD_addDeepBoostedJet_switch,
                                     jecPayload=nanoAOD_addDeepInfoAK8_switch.jecPayload)
    return process

def nanoAOD_customizeData(process):
    process = nanoAOD_customizeCommon(process)
    process = nanoAOD_recalibrateMETs(process,isData=True)
    for modifier in run2_nanoAOD_94XMiniAODv1, run2_nanoAOD_94XMiniAODv2:
        modifier.toModify(process, nanoAOD_runMETfixEE2017(process,isData=True))
    if hasattr(process,'calibratedPatElectrons80X'):
        process.calibratedPatElectrons80X.isMC = cms.bool(False)
        process.calibratedPatPhotons80X.isMC = cms.bool(False)
    return process

def nanoAOD_customizeMC(process):
    process = nanoAOD_customizeCommon(process)
    process = nanoAOD_recalibrateMETs(process,isData=False)
    for modifier in run2_nanoAOD_94XMiniAODv1, run2_nanoAOD_94XMiniAODv2:
        modifier.toModify(process, nanoAOD_runMETfixEE2017(process,isData=False))
    if hasattr(process,'calibratedPatElectrons80X'):
        process.calibratedPatElectrons80X.isMC = cms.bool(True)
        process.calibratedPatPhotons80X.isMC = cms.bool(True)
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
