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
from Configuration.Eras.Modifier_run2_nanoAOD_94X2016_cff import run2_nanoAOD_94X2016
from Configuration.Eras.Modifier_run2_nanoAOD_94XMiniAODv1_cff import run2_nanoAOD_94XMiniAODv1
from Configuration.Eras.Modifier_run2_nanoAOD_94XMiniAODv2_cff import run2_nanoAOD_94XMiniAODv2
from Configuration.Eras.Modifier_run2_nanoAOD_102Xv1_cff import run2_nanoAOD_102Xv1
from Configuration.Eras.Modifier_run2_nanoAOD_106Xv1_cff import run2_nanoAOD_106Xv1

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


genWeightsTable = cms.EDProducer("GenWeightsTableProducer",
    genEvent = cms.InputTag("generator"),
    genLumiInfoHeader = cms.InputTag("generator"),
    lheInfo = cms.VInputTag(cms.InputTag("externalLHEProducer"), cms.InputTag("source")),
    preferredPDFs = cms.VPSet( # see https://lhapdf.hepforge.org/pdfsets.html
        cms.PSet( name = cms.string("PDF4LHC15_nnlo_30_pdfas"), lhaid = cms.uint32(91400) ),
        cms.PSet( name = cms.string("NNPDF31_nnlo_hessian_pdfas"), lhaid = cms.uint32(306000) ),
        cms.PSet( name = cms.string("NNPDF30_nlo_as_0118"), lhaid = cms.uint32(260000) ), # for some 92X samples. Note that the nominal weight, 260000, is not included in the LHE ...
        cms.PSet( name = cms.string("NNPDF30_lo_as_0130"), lhaid = cms.uint32(262000) ), # some MLM 80X samples have only this (e.g. /store/mc/RunIISummer16MiniAODv2/DYJetsToLL_M-50_TuneCUETP8M1_13TeV-madgraphMLM-pythia8/MINIAODSIM/PUMoriond17_80X_mcRun2_asymptotic_2016_TrancheIV_v6_ext1-v2/120000/02A210D6-F5C3-E611-B570-008CFA197BD4.root )
        cms.PSet( name = cms.string("NNPDF30_nlo_nf_4_pdfas"), lhaid = cms.uint32(292000) ), # some FXFX 80X samples have only this (e.g. WWTo1L1Nu2Q, WWTo4Q)
        cms.PSet( name = cms.string("NNPDF30_nlo_nf_5_pdfas"), lhaid = cms.uint32(292200) ), # some FXFX 80X samples have only this (e.g. DYJetsToLL_Pt, WJetsToLNu_Pt, DYJetsToNuNu_Pt)
        cms.PSet( name = cms.string("NNPDF31_lo_as_0130"), lhaid = cms.uint32(315200) ), # SUSY signal samples use this
    ),
    namedWeightIDs = cms.vstring(),
    namedWeightLabels = cms.vstring(),
    lheWeightPrecision = cms.int32(14),
    maxPdfWeights = cms.uint32(150), 
    debug = cms.untracked.bool(False),
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
        nanoMetadata + jetSequence + muonSequence + tauSequence + electronSequence+photonSequence+vertexSequence+
        isoTrackSequence + jetLepSequence + # must be after all the leptons 
        linkedObjects  +
        jetTables + muonTables + tauTables + electronTables + photonTables +  globalTables +vertexTables+ metTables+simpleCleanerTable + isoTrackTables
        )
nanoSequenceOnlyFullSim = cms.Sequence(triggerObjectTables + l1bits)

nanoSequence = cms.Sequence(nanoSequenceCommon + nanoSequenceOnlyFullSim)

nanoSequenceFS = cms.Sequence(genParticleSequence + particleLevelSequence + nanoSequenceCommon + jetMC + muonMC + electronMC + photonMC + tauMC + metMC + ttbarCatMCProducers +  globalTablesMC + btagWeightTable + genWeightsTable + genParticleTables + particleLevelTables + lheInfoTable  + ttbarCategoryTable )

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

from PhysicsTools.PatUtils.tools.runMETCorrectionsAndUncertainties import runMetCorAndUncFromMiniAOD
#from PhysicsTools.PatAlgos.slimming.puppiForMET_cff import makePuppiesFromMiniAOD
def nanoAOD_recalibrateMETs(process,isData):
    runMetCorAndUncFromMiniAOD(process,isData=isData)
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
    process.electronSequence.insert(process.electronSequence.index(process.bitmapVIDForEle),process.egmGsfElectronIDSequence)
    for modifier in run2_miniAOD_80XLegacy, :
        modifier.toModify(process.electronMVAValueMapProducer, src = "slimmedElectronsUpdated")
        modifier.toModify(process.egmGsfElectronIDs, physicsObjectSrc = "slimmedElectronsUpdated")
    for modifier in run2_nanoAOD_94XMiniAODv1,run2_nanoAOD_94XMiniAODv2,run2_nanoAOD_94X2016 ,run2_nanoAOD_102Xv1:
        modifier.toModify(process.electronMVAValueMapProducer, src = "slimmedElectronsTo106X")
        modifier.toModify(process.egmGsfElectronIDs, physicsObjectSrc = "slimmedElectronsTo106X")
        
 

    switchOnVIDPhotonIdProducer(process,DataFormat.MiniAOD) # do not call this to avoid resetting photon IDs in VID, if called before inside makePuppiesFromMiniAOD
    for modname in photon_id_modules_WorkingPoints_nanoAOD.modules:
        setupAllVIDIdsInModule(process,modname,setupVIDPhotonSelection)
    process.photonSequence.insert(process.photonSequence.index(bitmapVIDForPho),process.egmPhotonIDSequence)
    for modifier in run2_miniAOD_80XLegacy,run2_nanoAOD_94XMiniAODv1,run2_nanoAOD_94XMiniAODv2,run2_nanoAOD_94X2016 ,run2_nanoAOD_102Xv1:
        modifier.toModify(process.photonMVAValueMapProducer, src = "slimmedPhotonsTo106X")
        modifier.toModify(process.egmPhotonIDs, physicsObjectSrc = "slimmedPhotonsTo106X")
    return process

def nanoAOD_addDeepInfoAK8(process,addDeepBTag,addDeepBoostedJet, addDeepDoubleX, jecPayload):
    _btagDiscriminators=[]
    if addDeepBTag:
        print("Updating process to run DeepCSV btag to AK8 jets")
        _btagDiscriminators += ['pfDeepCSVJetTags:probb','pfDeepCSVJetTags:probbb']
    if addDeepBoostedJet:
        print("Updating process to run DeepBoostedJet on datasets before 103X")
        from RecoBTag.ONNXRuntime.pfDeepBoostedJet_cff import _pfDeepBoostedJetTagsAll as pfDeepBoostedJetTagsAll
        _btagDiscriminators += pfDeepBoostedJetTagsAll
    if addDeepDoubleX: 
        print("Updating process to run DeepDoubleX on datasets before 104X")
        _btagDiscriminators += ['pfDeepDoubleBvLJetTags:probHbb', \
            'pfDeepDoubleCvLJetTags:probHcc', \
            'pfDeepDoubleCvBJetTags:probHcc', \
            'pfMassIndependentDeepDoubleBvLJetTags:probHbb', 'pfMassIndependentDeepDoubleCvLJetTags:probHcc', 'pfMassIndependentDeepDoubleCvBJetTags:probHcc']
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
        nanoAOD_addDeepDoubleX_switch = cms.untracked.bool(True), 
        jecPayload = cms.untracked.string('AK8PFPuppi')
        )
    # deepAK8 should not run on 80X, that contains ak8PFJetsCHS jets
    run2_miniAOD_80XLegacy.toModify(nanoAOD_addDeepInfoAK8_switch,
                                    nanoAOD_addDeepBTag_switch = cms.untracked.bool(True),
                                    nanoAOD_addDeepBoostedJet_switch = cms.untracked.bool(False),
                                    nanoAOD_addDeepDoubleX_switch = cms.untracked.bool(False),
                                    jecPayload = cms.untracked.string('AK8PFchs'))
    process = nanoAOD_addDeepInfoAK8(process,
                                     addDeepBTag=nanoAOD_addDeepInfoAK8_switch.nanoAOD_addDeepBTag_switch,
                                     addDeepBoostedJet=nanoAOD_addDeepInfoAK8_switch.nanoAOD_addDeepBoostedJet_switch,
                                     addDeepDoubleX=nanoAOD_addDeepInfoAK8_switch.nanoAOD_addDeepDoubleX_switch,
                                     jecPayload=nanoAOD_addDeepInfoAK8_switch.jecPayload)
    (run2_nanoAOD_94XMiniAODv1 | run2_nanoAOD_94X2016 | run2_nanoAOD_94XMiniAODv2 | run2_nanoAOD_102Xv1 | run2_nanoAOD_106Xv1).toModify(process, lambda p : nanoAOD_addTauIds(p))
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
