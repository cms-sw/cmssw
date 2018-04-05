
import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.Eras import eras

process = cms.Process('CTPPS2',eras.Run2_25ns)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
#process.load("CondFormats.CTPPSReadoutObjects.CTPPSPixelDAQMappingESSourceXML_cfi")

process.maxEvents = cms.untracked.PSet(
        input = cms.untracked.int32(1000)
        )

process.MessageLogger = cms.Service("MessageLogger",
    destinations = cms.untracked.vstring('cout'),
    cout = cms.untracked.PSet( threshold = cms.untracked.string('ERROR'))
)
process.source = cms.Source("EmptyIOVSource",
    timetype = cms.string('runnumber'),
    firstValue = cms.uint64(1),
    lastValue = cms.uint64(1),
    interval = cms.uint64(1)
)
process.SimpleMemoryCheck = cms.Service("SimpleMemoryCheck",ignoreTotal = cms.untracked.int32(1) )

process.source = cms.Source("NewEventStreamFileReader",
    fileNames = cms.untracked.vstring(
#'root://cms-xrd-global.cern.ch//store/data/Run2017A/ZeroBias2/AOD/PromptReco-v1/000/294/736/00000/44589413-F73F-E711-9E8D-02163E014337.root'
#'root://cms-xrd-global.cern.ch//store/data/Run2017B/ZeroBias/RAW/v1/000/299/065/00000/08980FD0-2D69-E711-98BC-02163E011B2D.root'
#'root://eoscms.cern.ch//eos/cms/store/t0streamer/Data/HLTMonitor/000/300/742/run300742_ls0124_streamHLTMonitor_StorageManager.dat'
'root://eoscms.cern.ch//eos/cms/store/t0streamer/Data/HLTMonitor/000/300/811/run300811_ls0024_streamHLTMonitor_StorageManager.dat'
),
#duplicateCheckMode = cms.untracked.string("checkEachFile")
firstEvent = cms.untracked.uint64(1)
#141309767)
)



from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_hlt_relval', '')

# raw-to-digi conversion
process.load("EventFilter.CTPPSRawToDigi.ctppsRawToDigi_cff")

process.load("RecoCTPPS.Configuration.recoCTPPS_cff")


############
process.o1 = cms.OutputModule("PoolOutputModule",
        outputCommands = cms.untracked.vstring('drop *',
                                               'keep *_totemRP*_*_*',
                                               'keep *_ctpps*_*_*'

),
        fileName = cms.untracked.string('simevent_CTPPS_CLU_REC_real_mem_testDB_300811.root')
        )


process.mixedigi_step = cms.Path(
process.ctppsRawToDigi
*process.recoCTPPS

)

process.outpath = cms.EndPath(process.o1)

process.schedule = cms.Schedule(process.mixedigi_step,process.outpath)

# filter all path with the production filter sequence
for path in process.paths:
  #  getattr(process,path)._seq = process.ProductionFilterSequence * getattr(process,path)._seq
    getattr(process,path)._seq =  getattr(process,path)._seq


