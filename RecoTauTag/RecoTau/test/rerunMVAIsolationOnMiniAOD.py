import FWCore.ParameterSet.Config as cms

process = cms.Process("rerunMVAIsolationOnMiniAOD")

## Geometry and Detector Conditions (needed for a few tau reco steps)
process.load("Configuration.Geometry.GeometryRecoDB_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc')
process.load("Configuration.StandardSequences.MagneticField_cff")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 1
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.source = cms.Source("PoolSource",
    # replace 'myfile.root' with the source file you want to use
    fileNames = cms.untracked.vstring(
        'file:patMiniAOD_standard.root'
    )
)

process.TFileService = cms.Service("TFileService",
  fileName = cms.string('rerunTauID_miniAOD.root')
)

#from Configuration.AlCa.GlobalTag import GlobalTag
#process.GlobalTag = GlobalTag(process.GlobalTag, '76X_mcRun2_asymptotic_RunIIFall15DR76_v1', '')

from RecoTauTag.RecoTau.TauDiscriminatorTools import noPrediscriminants
process.load('RecoTauTag.Configuration.loadRecoTauTagMVAsFromPrepDB_cfi')
from RecoTauTag.RecoTau.PATTauDiscriminationByMVAIsolationRun2_cff import *
from RecoTauTag.RecoTau.PATTauDiscriminationAgainstElectronMVA6_cfi import *

process.rerunDiscriminationByIsolationMVArun2v1raw = patDiscriminationByIsolationMVArun2v1raw.clone(
    PATTauProducer = cms.InputTag('slimmedTaus'),
    Prediscriminants = noPrediscriminants,
    loadMVAfromDB = cms.bool(True),
    ## run with old MVA training
    #mvaName = cms.string("RecoTauTag_tauIdMVADBoldDMwLTv1"),
    ## run with new MVA training
    mvaName = cms.string("RecoTauTag_tauIdMVAIsoDBoldDMwLT2016v1"),
    mvaOpt = cms.string("DBoldDMwLT"),
    requireDecayMode = cms.bool(True),
    verbosity = cms.int32(0)
)

process.rerunDiscriminationByIsolationMVArun2v1VLoose = patDiscriminationByIsolationMVArun2v1VLoose.clone(
    PATTauProducer = cms.InputTag('slimmedTaus'),    
    Prediscriminants = noPrediscriminants,
    toMultiplex = cms.InputTag('rerunDiscriminationByIsolationMVArun2v1raw'),
    key = cms.InputTag('rerunDiscriminationByIsolationMVArun2v1raw:category'),
    loadMVAfromDB = cms.bool(True),
    mvaOutput_normalization = cms.string("RecoTauTag_tauIdMVADBoldDMwLTv1_mvaOutput_normalization"),
    mapping = cms.VPSet(
        cms.PSet(
            category = cms.uint32(0),
            cut = cms.string("RecoTauTag_tauIdMVADBoldDMwLTv1_WPEff90"),
            variable = cms.string("pt"),
        )
    )
)

process.rerunDiscriminationAgainstElectronMVA6 = patTauDiscriminationAgainstElectronMVA6.clone(
    PATTauProducer = cms.InputTag('slimmedTaus'),
    Prediscriminants = noPrediscriminants,
    #Prediscriminants = requireLeadTrack,
    loadMVAfromDB = cms.bool(True),
    returnMVA = cms.bool(True),
    method = cms.string("BDTG"),
    mvaName_NoEleMatch_woGwoGSF_BL = cms.string("RecoTauTag_antiElectronMVA6v1_gbr_NoEleMatch_woGwoGSF_BL"),
    mvaName_NoEleMatch_wGwoGSF_BL = cms.string("RecoTauTag_antiElectronMVA6v1_gbr_NoEleMatch_wGwoGSF_BL"),
    mvaName_woGwGSF_BL = cms.string("RecoTauTag_antiElectronMVA6v1_gbr_woGwGSF_BL"),
    mvaName_wGwGSF_BL = cms.string("RecoTauTag_antiElectronMVA6v1_gbr_wGwGSF_BL"),
    mvaName_NoEleMatch_woGwoGSF_EC = cms.string("RecoTauTag_antiElectronMVA6v1_gbr_NoEleMatch_woGwoGSF_EC"),
    mvaName_NoEleMatch_wGwoGSF_EC = cms.string("RecoTauTag_antiElectronMVA6v1_gbr_NoEleMatch_wGwoGSF_EC"),
    mvaName_woGwGSF_EC = cms.string("RecoTauTag_antiElectronMVA6v1_gbr_woGwGSF_EC"),
    mvaName_wGwGSF_EC = cms.string("RecoTauTag_antiElectronMVA6v1_gbr_wGwGSF_EC"),
    minMVANoEleMatchWOgWOgsfBL = cms.double(0.0),
    minMVANoEleMatchWgWOgsfBL  = cms.double(0.0),
    minMVAWOgWgsfBL            = cms.double(0.0),
    minMVAWgWgsfBL             = cms.double(0.0),
    minMVANoEleMatchWOgWOgsfEC = cms.double(0.0),
    minMVANoEleMatchWgWOgsfEC  = cms.double(0.0),
    minMVAWOgWgsfEC            = cms.double(0.0),
    minMVAWgWgsfEC             = cms.double(0.0),
    srcElectrons = cms.InputTag('slimmedElectrons'),
    usePhiAtEcalEntranceExtrapolation = cms.bool(True)
)

process.rerunDiscriminationByIsolationMVArun2v1Loose = process.rerunDiscriminationByIsolationMVArun2v1VLoose.clone()
process.rerunDiscriminationByIsolationMVArun2v1Loose.mapping[0].cut = cms.string("RecoTauTag_tauIdMVADBoldDMwLTv1_WPEff80")
process.rerunDiscriminationByIsolationMVArun2v1Medium = process.rerunDiscriminationByIsolationMVArun2v1VLoose.clone()
process.rerunDiscriminationByIsolationMVArun2v1Medium.mapping[0].cut = cms.string("RecoTauTag_tauIdMVADBoldDMwLTv1_WPEff70")
process.rerunDiscriminationByIsolationMVArun2v1Tight = process.rerunDiscriminationByIsolationMVArun2v1VLoose.clone()
process.rerunDiscriminationByIsolationMVArun2v1Tight.mapping[0].cut = cms.string("RecoTauTag_tauIdMVADBoldDMwLTv1_WPEff60")
process.rerunDiscriminationByIsolationMVArun2v1VTight = process.rerunDiscriminationByIsolationMVArun2v1VLoose.clone()
process.rerunDiscriminationByIsolationMVArun2v1VTight.mapping[0].cut = cms.string("RecoTauTag_tauIdMVADBoldDMwLTv1_WPEff50")
process.rerunDiscriminationByIsolationMVArun2v1VVTight = process.rerunDiscriminationByIsolationMVArun2v1VLoose.clone()
process.rerunDiscriminationByIsolationMVArun2v1VVTight.mapping[0].cut = cms.string("RecoTauTag_tauIdMVADBoldDMwLTv1_WPEff40")

process.rerunMvaIsolation2SeqRun2 = cms.Sequence(
   process.rerunDiscriminationByIsolationMVArun2v1raw
   *process.rerunDiscriminationByIsolationMVArun2v1VLoose
   *process.rerunDiscriminationByIsolationMVArun2v1Loose
   *process.rerunDiscriminationByIsolationMVArun2v1Medium
   *process.rerunDiscriminationByIsolationMVArun2v1Tight
   *process.rerunDiscriminationByIsolationMVArun2v1VTight
   *process.rerunDiscriminationByIsolationMVArun2v1VVTight
)

process.rerunMVAIsolationOnMiniAOD = cms.EDAnalyzer('rerunMVAIsolationOnMiniAOD'
)

process.rerunMVAIsolationOnMiniAOD.verbosity = cms.int32(0)
process.rerunMVAIsolationOnMiniAOD.additionalCollectionsAvailable = cms.bool(True)

# embed new id's into tau
embedID = cms.EDProducer("PATTauIDEmbedder",
   src = cms.InputTag('slimmedTaus'),
   tauIDSources = cms.PSet(
      byIsolationMVArun2v1DBoldDMwLTrawNew = cms.InputTag('rerunDiscriminationByIsolationMVArun2v1raw'),
      byVLooseIsolationMVArun2v1DBoldDMwLTNew = cms.InputTag('rerunDiscriminationByIsolationMVArun2v1VLoose'),
      byLooseIsolationMVArun2v1DBoldDMwLTNew = cms.InputTag('rerunDiscriminationByIsolationMVArun2v1Loose'),
      byMediumIsolationMVArun2v1DBoldDMwLTNew = cms.InputTag('rerunDiscriminationByIsolationMVArun2v1Medium'),
      byTightIsolationMVArun2v1DBoldDMwLTNew = cms.InputTag('rerunDiscriminationByIsolationMVArun2v1Tight'),
      byVTightIsolationMVArun2v1DBoldDMwLTNew = cms.InputTag('rerunDiscriminationByIsolationMVArun2v1VTight'),
      byVVTightIsolationMVArun2v1DBoldDMwLTNew = cms.InputTag('rerunDiscriminationByIsolationMVArun2v1VVTight'),
      againstElectronMVA6RawNew = cms.InputTag('rerunDiscriminationAgainstElectronMVA6')
   ),
)
setattr(process, "newTauIDsEmbedded", embedID)

## added for mvaIsolation on miniAOD testing
process.out = cms.OutputModule("PoolOutputModule",
                               fileName = cms.untracked.string('patTuple_newTauIDs.root'),
                               ## save only events passing the full path
                               #SelectEvents = cms.untracked.PSet( SelectEvents = cms.vstring('p') ),
                               ## save PAT output; you need a '*' to unpack the list of commands
                               ## 'patEventContent'
                               outputCommands = cms.untracked.vstring('drop *', "keep *_newTauIDsEmbedded_*_*")
                               )

process.p = cms.Path(
   process.rerunMvaIsolation2SeqRun2
  *process.rerunDiscriminationAgainstElectronMVA6
  *process.newTauIDsEmbedded
  *process.rerunMVAIsolationOnMiniAOD
)

process.outpath = cms.EndPath(process.out)
