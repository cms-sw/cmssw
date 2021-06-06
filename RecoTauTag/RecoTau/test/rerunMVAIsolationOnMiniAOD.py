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
from RecoTauTag.RecoTau.patTauDiscriminationAgainstElectronMVA6_cfi import *

process.rerunDiscriminationByIsolationMVArun2v1raw = patDiscriminationByIsolationMVArun2v1raw.clone(
    PATTauProducer = cms.InputTag('slimmedTaus'),
    Prediscriminants = noPrediscriminants,
    loadMVAfromDB = cms.bool(True),
    ## run with old MVA training
    #mvaName = cms.string("RecoTauTag_tauIdMVADBoldDMwLTv1"),
    ## run with new MVA training
    mvaName = cms.string("RecoTauTag_tauIdMVAIsoDBoldDMwLT2016v1"),
    mvaOpt = cms.string("DBoldDMwLT"),
    verbosity = cms.int32(0)
)

process.rerunDiscriminationByIsolationMVArun2v1 = patDiscriminationByIsolationMVArun2v1.clone(
    PATTauProducer = cms.InputTag('slimmedTaus'),    
    Prediscriminants = noPrediscriminants,
    toMultiplex = cms.InputTag('rerunDiscriminationByIsolationMVArun2v1raw'),
    loadMVAfromDB = cms.bool(True),
    mvaOutput_normalization = cms.string("RecoTauTag_tauIdMVADBoldDMwLTv1_mvaOutput_normalization"),
    mapping = cms.VPSet(
        cms.PSet(
            category = cms.uint32(0),
            cut = cms.string("RecoTauTag_tauIdMVADBoldDMwLTv1"),
            variable = cms.string("pt"),
        )
    ),
    workingPoints = cms.vstring(
        "_WPEff90",
        "_WPEff80",
        "_WPEff70",
        "_WPEff60",
        "_WPEff50",
        "_WPEff40"
    )
)

process.rerunDiscriminationAgainstElectronMVA6 = patTauDiscriminationAgainstElectronMVA6.clone(
    PATTauProducer = cms.InputTag('slimmedTaus'),
    Prediscriminants = noPrediscriminants,
    #Prediscriminants = requireLeadTrack,
    srcElectrons = cms.InputTag('slimmedElectrons'),
    loadMVAfromDB = cms.bool(True),
    vetoEcalCracks = cms.bool(False),
    returnMVA = cms.bool(True),
    method = cms.string("BDTG"),
    mvaName_NoEleMatch_woGwoGSF_BL = cms.string("RecoTauTag_antiElectronMVA6v3_noeveto_gbr_NoEleMatch_woGwoGSF_BL"),
    mvaName_NoEleMatch_wGwoGSF_BL = cms.string("RecoTauTag_antiElectronMVA6v3_noeveto_gbr_NoEleMatch_wGwoGSF_BL"),
    mvaName_woGwGSF_BL = cms.string("RecoTauTag_antiElectronMVA6v3_noeveto_gbr_woGwGSF_BL"),
    mvaName_wGwGSF_BL = cms.string("RecoTauTag_antiElectronMVA6v3_noeveto_gbr_wGwGSF_BL"),
    mvaName_NoEleMatch_woGwoGSF_EC = cms.string("RecoTauTag_antiElectronMVA6v3_noeveto_gbr_NoEleMatch_woGwoGSF_EC"),
    mvaName_NoEleMatch_wGwoGSF_EC = cms.string("RecoTauTag_antiElectronMVA6v3_noeveto_gbr_NoEleMatch_wGwoGSF_EC"),
    mvaName_woGwGSF_EC = cms.string("RecoTauTag_antiElectronMVA6v3_noeveto_gbr_woGwGSF_EC"),
    mvaName_wGwGSF_EC = cms.string("RecoTauTag_antiElectronMVA6v3_noeveto_gbr_wGwGSF_EC"),
    minMVANoEleMatchWOgWOgsfBL = cms.double(0.0),
    minMVANoEleMatchWgWOgsfBL  = cms.double(0.0),
    minMVAWOgWgsfBL            = cms.double(0.0),
    minMVAWgWgsfBL             = cms.double(0.0),
    minMVANoEleMatchWOgWOgsfEC = cms.double(0.0),
    minMVANoEleMatchWgWOgsfEC  = cms.double(0.0),
    minMVAWOgWgsfEC            = cms.double(0.0),
    minMVAWgWgsfEC             = cms.double(0.0),
    usePhiAtEcalEntranceExtrapolation = cms.bool(True)
)

process.rerunMvaIsolation2SeqRun2 = cms.Sequence(
   process.rerunDiscriminationByIsolationMVArun2v1raw
   *process.rerunDiscriminationByIsolationMVArun2v1
)

process.rerunMVAIsolationOnMiniAOD = cms.EDAnalyzer('rerunMVAIsolationOnMiniAOD'
)

process.rerunMVAIsolationOnMiniAOD.verbosity = cms.int32(0)
process.rerunMVAIsolationOnMiniAOD.additionalCollectionsAvailable = cms.bool(True)

# embed new id's into tau
def tauIDMVAinputs(module, wp):
    return cms.PSet(inputTag = cms.InputTag(module), workingPointIndex = cms.int32(-1 if wp=="raw" else -2 if wp=="category" else getattr(process, module).workingPoints.index(wp)))
embedID = cms.EDProducer("PATTauIDEmbedder",
   src = cms.InputTag('slimmedTaus'),
   tauIDSources = cms.PSet(
      byIsolationMVArun2v1DBoldDMwLTrawNew = tauIDMVAinputs("rerunDiscriminationByIsolationMVArun2v1", "raw"),
      byVLooseIsolationMVArun2v1DBoldDMwLTNew = tauIDMVAinputs("rerunDiscriminationByIsolationMVArun2v1", "_WPEff90"),
      byLooseIsolationMVArun2v1DBoldDMwLTNew = tauIDMVAinputs("rerunDiscriminationByIsolationMVArun2v1", "_WPEff80"),
      byMediumIsolationMVArun2v1DBoldDMwLTNew = tauIDMVAinputs("rerunDiscriminationByIsolationMVArun2v1", "_WPEff70"),
      byTightIsolationMVArun2v1DBoldDMwLTNew = tauIDMVAinputs("rerunDiscriminationByIsolationMVArun2v1", "_WPEff60"),
      byVTightIsolationMVArun2v1DBoldDMwLTNew = tauIDMVAinputs("rerunDiscriminationByIsolationMVArun2v1", "_WPEff50"),
      byVVTightIsolationMVArun2v1DBoldDMwLTNew = tauIDMVAinputs("rerunDiscriminationByIsolationMVArun2v1", "_WPEff40"),
      againstElectronMVA6RawNew = tauIDMVAinputs("rerunDiscriminationAgainstElectronMVA6", "raw")
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
