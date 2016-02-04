import FWCore.ParameterSet.Config as cms

process = cms.Process("TTRIGVALIDPROC")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.threshold = 'WARNING'

process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = ''

process.load("CondCore.DBCommon.CondDBSetup_cfi")
process.load("DQMServices.Core.DQM_cfg")
#process.load("RecoLocalMuon.Configuration.RecoLocalMuonCosmics_cff")
process.load("RecoLocalMuon.Configuration.RecoLocalMuon_cff")

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring()
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

"""
process.calibDB = cms.ESSource("PoolDBESSource",
    process.CondDBSetup,
    timetype = cms.string('runnumber'),
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('DTTtrigRcd'),
        tag = cms.string('ttrig')
    )),
                          
    connect = cms.string('sqlite_file:'),
    authenticationMethod = cms.untracked.uint32(0)
)
process.es_prefer_calibDB = cms.ESPrefer('PoolDBESSource','calibDB')
"""

# if read from RAW
#process.load("EventFilter.DTRawToDigi.dtunpacker_cfi")

process.eventInfoProvider = cms.EDFilter("EventCoordinatesSource",
    eventInfoFolder = cms.untracked.string('EventInfo/')
)

process.DTkFactValidation = cms.EDAnalyzer("DTCalibValidation",
    # Write the histos on file
    OutputMEsInRootFile = cms.bool(True),
    # Lable to retrieve 2D segments from the event
    segment2DLabel = cms.untracked.string('dt2DSegments'),
    OutputFileName = cms.string('residuals.root'),
    # Lable to retrieve 4D segments from the event
    segment4DLabel = cms.untracked.string('dt4DSegments'),
    debug = cms.untracked.bool(False),
    # Lable to retrieve RecHits from the event
    recHits1DLabel = cms.untracked.string('dt1DRecHits'),
    # Detailed analysis
    detailedAnalysis = cms.untracked.bool(False)
)

process.output = cms.OutputModule("PoolOutputModule",
                  outputCommands = cms.untracked.vstring(
                      'drop *', 
                      'keep *_MEtoEDMConverter_*_*'),
                  fileName = cms.untracked.string('DQM.root')
                  #SelectEvents = cms.untracked.PSet(
                  #    SelectEvents = cms.vstring('analysis_step')
                  #)
)
process.load("DQMServices.Components.MEtoEDMConverter_cff")
#process.dummyProducer = cms.EDProducer("ThingWithMergeProducer")

# if read from RAW
#process.firstStep = cms.Sequence(process.muonDTDigis*process.dt1DRecHits*process.dt2DSegments*process.dt4DSegments*process.DTkFactValidation)
#process.firstStep = cms.Sequence(process.dummyProducer + process.muonDTDigis*process.dt1DRecHits*process.dt2DSegments*process.dt4DSegments*process.DTkFactValidation*process.MEtoEDMConverter)
#process.firstStep = cms.Sequence(process.dummyProducer + process.dt1DRecHits*process.dt2DSegments*process.dt4DSegments*process.DTkFactValidation*process.MEtoEDMConverter)

process.dtValidSequence = cms.Sequence(process.dt1DRecHits*process.dt2DSegments*process.dt4DSegments*process.DTkFactValidation)

process.analysis_step = cms.Path(process.dtValidSequence*process.MEtoEDMConverter)
process.out_step = cms.EndPath(process.output)
process.DQM.collectorHost = ''
