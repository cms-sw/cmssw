import FWCore.ParameterSet.Config as cms
process = cms.Process('ANALYSIS')

process.load('Configuration.StandardSequences.Services_cff')
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag='92X_dataRun2_Prompt_v4'

process.MessageLogger = cms.Service("MessageLogger",
    destinations = cms.untracked.vstring('warnings','errors',
                                         'cout','cerr'),
    categories = cms.untracked.vstring('HcalIsoTrack'), 
    debugModules = cms.untracked.vstring('*'),
    warnings = cms.untracked.PSet(
        placeholder = cms.untracked.bool(True)
    ),
    default = cms.untracked.PSet(

    ),
    errors = cms.untracked.PSet(
        placeholder = cms.untracked.bool(True)
    ),
    cerr = cms.untracked.PSet(
        optionalPSet = cms.untracked.bool(True),
        INFO = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        noTimeStamps = cms.untracked.bool(False),
        FwkReport = cms.untracked.PSet(
            optionalPSet = cms.untracked.bool(True),
            reportEvery = cms.untracked.int32(500),
            limit = cms.untracked.int32(10000000)
        ),
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        Root_NoDictionary = cms.untracked.PSet(
            optionalPSet = cms.untracked.bool(True),
            limit = cms.untracked.int32(0)
        ),
        FwkJob = cms.untracked.PSet(
            optionalPSet = cms.untracked.bool(True),
            limit = cms.untracked.int32(0)
        ),
        FwkSummary = cms.untracked.PSet(
            optionalPSet = cms.untracked.bool(True),
            reportEvery = cms.untracked.int32(1),
            limit = cms.untracked.int32(10000000)
        ),
        threshold = cms.untracked.string('INFO')
     ),
     cout = cms.untracked.PSet(
        threshold = cms.untracked.string('INFO'),
        noTimeStamps = cms.untracked.bool(True),
        INFO = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        DEBUG = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        HcalIsoTrack = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
       )
    )
)

process.load('RecoLocalCalo.CaloTowersCreator.calotowermaker_cfi')
process.towerMakerAll = process.calotowermaker.clone()
process.towerMakerAll.hbheInput = cms.InputTag("hbhereco")
process.towerMakerAll.hoInput = cms.InputTag("none")
process.towerMakerAll.hfInput = cms.InputTag("none")
process.towerMakerAll.ecalInputs = cms.VInputTag(cms.InputTag("ecalRecHit","EcalRecHitsEB"), cms.InputTag("ecalRecHit","EcalRecHitsEE"))
process.towerMakerAll.AllowMissingInputs = True

process.load('Calibration.HcalCalibAlgos.HcalIsoTrkAnalyzer_cfi')
process.source = cms.Source("PoolSource", 
                            fileNames = cms.untracked.vstring(
        '/store/data/Run2017A/JetHT/RECO/PromptReco-v2/000/296/168/00000/CE1DFEC8-444C-E711-AB09-02163E019CB1.root'
    )
)

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )
process.MessageLogger.cerr.FwkReport.reportEvery = 1
process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )
process.TFileService = cms.Service("TFileService",
   fileName = cms.string('output.root')
)
process.HcalIsoTrkAnalyzer.Triggers = []
process.p = cms.Path(process.HcalIsoTrkAnalyzer)

