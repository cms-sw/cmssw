import FWCore.ParameterSet.Config as cms

process = cms.Process("PROD")

process.MessageLogger = cms.Service("MessageLogger",
    debugModules = cms.untracked.vstring('resolutionTest_step1',
        'resolutionTest_step2',
        'resolutionTest_step3'),
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('ERROR'),
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        resolution = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        noLineBreaks = cms.untracked.bool(True)
    ),
    categories = cms.untracked.vstring('resolution'),
    destinations = cms.untracked.vstring('cout')
)

process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = "GLOBALTAGTEMPLATE"

process.load("CondCore.DBCommon.CondDBSetup_cfi")

process.load("DQMServices.Core.DQM_cfg")

process.load("RecoLocalMuon.Configuration.RecoLocalMuonCosmics_cff")

process.source = cms.Source("PoolSource",
    debugFlag = cms.untracked.bool(True),
    debugVebosity = cms.untracked.uint32(10),
    fileNames = cms.untracked.vstring()
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.calibDB = cms.ESSource("PoolDBESSource",
    process.CondDBSetup,
    timetype = cms.string('runnumber'),
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('DTTtrigRcd'),
        tag = cms.string('ttrig')
    )),
    connect = cms.string('sqlite_file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_MUONCALIB/DTCALIB/RUNPERIODTEMPLATE/ttrig/ttrig_DUMPDBTEMPL_RUNNUMBERTEMPLATE.db'),
    authenticationMethod = cms.untracked.uint32(0)
)
process.es_prefer_calibDB = cms.ESPrefer('PoolDBESSource','calibDB')

# if read from RAW
process.load("EventFilter.DTRawToDigi.dtunpacker_cfi")

process.eventInfoProvider = cms.EDFilter("EventCoordinatesSource",
    eventInfoFolder = cms.untracked.string('EventInfo/')
)

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
process.DTkFactValidation = DQMEDAnalyzer('DTCalibValidation',
    # Write the histos on file
    OutputMEsInRootFile = cms.bool(True),
    # Lable to retrieve 2D segments from the event
    segment2DLabel = cms.untracked.string('dt2DSegments'),
    OutputFileName = cms.string('residuals.root'),
    # Lable to retrieve 4D segments from the event
    segment4DLabel = cms.untracked.string('dt4DSegments'),
    debug = cms.untracked.bool(False),
    # Lable to retrieve RecHits from the event
    recHits1DLabel = cms.untracked.string('dt1DRecHits')
)

process.FEVT = cms.OutputModule("PoolOutputModule",
               outputCommands = cms.untracked.vstring('drop *', 
                                'keep *_MEtoEDMConverter_*_*'),
               fileName = cms.untracked.string('DQM.root')
                               )
process.load("DQMServices.Components.MEtoEDMConverter_cff")
process.dummyProducer = cms.EDProducer("ThingWithMergeProducer")

# if read from RAW
#process.firstStep = cms.Sequence(process.muonDTDigis*process.dt1DRecHits*process.dt2DSegments*process.dt4DSegments*process.DTkFactValidation)
process.firstStep = cms.Sequence(process.dummyProducer + process.muonDTDigis*process.dt1DRecHits*process.dt2DSegments*process.dt4DSegments*process.DTkFactValidation*process.MEtoEDMConverter)

#process.firstStep = cms.Sequence(process.dummyProducer + process.dt1DRecHits*process.dt2DSegments*process.dt4DSegments*process.DTkFactValidation*process.MEtoEDMConverter)
process.p = cms.Path(process.firstStep)
process.outpath = cms.EndPath(process.FEVT)
process.DQM.collectorHost = ''
