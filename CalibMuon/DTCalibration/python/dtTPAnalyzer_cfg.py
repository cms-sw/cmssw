import FWCore.ParameterSet.Config as cms

process = cms.Process("DTTPAnalyzer")

process.load("FWCore.MessageService.MessageLogger_cfi")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag=autoCond['run3_data']
process.load("Configuration.StandardSequences.GeometryDB_cff")
process.load("Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff")

process.load("CondCore.CondDB.CondDB_cfi")

process.load("DQMServices.Core.DQM_cfg")

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring()
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )

process.dtunpacker = cms.EDProducer("DTUnpackingModule",
    dataType = cms.string('DDU'),
    inputLabel = cms.InputTag('source'),
    fedbyType = cms.bool(False),
    useStandardFEDid = cms.bool(True),
    dqmOnly = cms.bool(False),                       
    readOutParameters = cms.PSet(
        debug = cms.untracked.bool(False),
        rosParameters = cms.PSet(
            writeSC = cms.untracked.bool(True),
            readingDDU = cms.untracked.bool(True),
            performDataIntegrityMonitor = cms.untracked.bool(False),
            readDDUIDfromDDU = cms.untracked.bool(True),
            debug = cms.untracked.bool(False),
            localDAQ = cms.untracked.bool(False)
        ),
        localDAQ = cms.untracked.bool(False),
        performDataIntegrityMonitor = cms.untracked.bool(False)
    )
)

process.dtTPAnalyzer = cms.EDAnalyzer("DTTPAnalyzer",
    digiLabel = cms.InputTag('dtunpacker'),
    rootFileName = cms.untracked.string(''),
    subtractT0 = cms.bool(True),
    tTrigMode = cms.string('DTTTrigSyncT0Only'),
    tTrigModeConfig = cms.PSet(debug = cms.untracked.bool(False))
)

process.p = cms.Path(process.dtunpacker*process.dtTPAnalyzer)
