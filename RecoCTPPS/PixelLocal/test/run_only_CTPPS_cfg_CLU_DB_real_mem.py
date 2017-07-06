import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.Eras import eras

process = cms.Process('CTPPS2',eras.Run2_25ns)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.maxEvents = cms.untracked.PSet(
        input = cms.untracked.int32(10000)
        )

process.MessageLogger = cms.Service("MessageLogger",
    destinations = cms.untracked.vstring('cclu_info'),
    cclu_info = cms.untracked.PSet( threshold = cms.untracked.string('INFO'))
)
process.source = cms.Source("EmptyIOVSource",
    timetype = cms.string('runnumber'),
    firstValue = cms.uint64(1),
    lastValue = cms.uint64(1),
    interval = cms.uint64(1)
)

# Track memory leaks
process.SimpleMemoryCheck = cms.Service("SimpleMemoryCheck",ignoreTotal = cms.untracked.int32(1) )


process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
'root://cms-xrd-global.cern.ch//store/data/Run2017A/ZeroBias2/AOD/PromptReco-v1/000/294/736/00000/44589413-F73F-E711-9E8D-02163E014337.root'

),
duplicateCheckMode = cms.untracked.string("checkEachFile")
)


from CondCore.CondDB.CondDB_cfi import *
process.PoolDBESSource = cms.ESSource("PoolDBESSource",
        CondDB.clone(
connect = cms.string('frontier://FrontierPrep/CMS_CONDITIONS')),
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('CTPPSPixelGainCalibrationsRcd'),
        tag = cms.string("CTPPSPixelGainCalibNew_v4_test")
    ),
      cms.PSet(
        record = cms.string('CTPPSPixelDAQMappingRcd'),
        tag = cms.string("CTPPSPixelDAQMapping_v1_test"),
        label = cms.untracked.string("RPix")
      ),
      cms.PSet(
        record = cms.string('CTPPSPixelAnalysisMaskRcd'),
        tag = cms.string("CTPPSPixelAnalysisMask_v1_test"),
        label = cms.untracked.string("RPix")
      )


)
)



# raw-to-digi conversion
process.load("EventFilter.CTPPSRawToDigi.ctppsRawToDigi_cff")

############
process.o1 = cms.OutputModule("PoolOutputModule",
        outputCommands = cms.untracked.vstring('drop *',
                                               'keep CTPPSPixelClusteredmDetSetVector_ctppsPixelClusters_*_*',
                                               'keep CTPPSPixelRecHitedmDetSetVector_recHitProd_*_*',
),
        fileName = cms.untracked.string('simevent_CTPPS_CLU_DB_real_mem.root')
        )

process.load("RecoCTPPS.Configuration.recoCTPPS_cff")


process.ALL = cms.Path(process.recoCTPPS)


process.outpath = cms.EndPath(process.o1)

process.schedule = cms.Schedule(process.ALL,process.outpath)

# filter all path with the production filter sequence
for path in process.paths:
  #  getattr(process,path)._seq = process.ProductionFilterSequence * getattr(process,path)._seq
    getattr(process,path)._seq =  getattr(process,path)._seq


