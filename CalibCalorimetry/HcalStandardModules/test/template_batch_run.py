import FWCore.ParameterSet.Config as cms

process = cms.Process('peds')
process.load('CondCore.DBCommon.CondDBSetup_cfi')
process.load('Configuration.StandardSequences.RawToDigi_Data_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.GlobalTag.globaltag = cms.string('GR_R_311_V1::All')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.MessageLogger.cerr.FwkReport.reportEvery = 10000

process.maxEvents = cms.untracked.PSet(
#    input = cms.untracked.int32(150000)
    input = cms.untracked.int32(-1)
)

process.source = cms.Source('PoolSource',
    fileNames = cms.untracked.vstring(
 #INPUTFILES
)
)

process.filt = cms.EDFilter('HcalCalibTypeFilter',
    InputLabel    = cms.string( 'source' ),
    CalibTypes    = cms.vint32( 1 ), # 2,3,4,5
    FilterSummary = cms.untracked.bool( True )
)

process.analyzepeds = cms.EDAnalyzer('HcalPedestalsAnalysis',
    dumpXML = cms.untracked.bool(False)
)

process.p = cms.Path(process.filt*process.hcalDigis*process.analyzepeds)
