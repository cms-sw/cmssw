import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.Eras import eras

process = cms.Process('CTPPS2',eras.Run2_25ns)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.maxEvents = cms.untracked.PSet(
        input = cms.untracked.int32(200)
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

process.source = cms.Source("NewEventStreamFileReader",
    fileNames = cms.untracked.vstring(
'file:/eos/cms/store/t0streamer/Data/Express/000/293/910/run293910_ls0310_streamExpress_StorageManager.dat',
'file:/eos/cms/store/t0streamer/Data/Express/000/293/910/run293910_ls0311_streamExpress_StorageManager.dat',
'file:/eos/cms/store/t0streamer/Data/Express/000/293/910/run293910_ls0312_streamExpress_StorageManager.dat',
'file:/eos/cms/store/t0streamer/Data/Express/000/293/910/run293910_ls0313_streamExpress_StorageManager.dat',
'file:/eos/cms/store/t0streamer/Data/Express/000/293/910/run293910_ls0314_streamExpress_StorageManager.dat',
'file:/eos/cms/store/t0streamer/Data/Express/000/293/910/run293910_ls0315_streamExpress_StorageManager.dat',
'file:/eos/cms/store/t0streamer/Data/Express/000/293/910/run293910_ls0316_streamExpress_StorageManager.dat',
'file:/eos/cms/store/t0streamer/Data/Express/000/293/910/run293910_ls0317_streamExpress_StorageManager.dat',
'file:/eos/cms/store/t0streamer/Data/Express/000/293/910/run293910_ls0318_streamExpress_StorageManager.dat',
'file:/eos/cms/store/t0streamer/Data/Express/000/293/910/run293910_ls0319_streamExpress_StorageManager.dat',

'file:/eos/cms/store/t0streamer/Data/Express/000/293/910/run293910_ls0320_streamExpress_StorageManager.dat',
'file:/eos/cms/store/t0streamer/Data/Express/000/293/910/run293910_ls0321_streamExpress_StorageManager.dat',
'file:/eos/cms/store/t0streamer/Data/Express/000/293/910/run293910_ls0322_streamExpress_StorageManager.dat',
'file:/eos/cms/store/t0streamer/Data/Express/000/293/910/run293910_ls0323_streamExpress_StorageManager.dat',
'file:/eos/cms/store/t0streamer/Data/Express/000/293/910/run293910_ls0324_streamExpress_StorageManager.dat',
'file:/eos/cms/store/t0streamer/Data/Express/000/293/910/run293910_ls0325_streamExpress_StorageManager.dat',
'file:/eos/cms/store/t0streamer/Data/Express/000/293/910/run293910_ls0326_streamExpress_StorageManager.dat',
'file:/eos/cms/store/t0streamer/Data/Express/000/293/910/run293910_ls0327_streamExpress_StorageManager.dat',
'file:/eos/cms/store/t0streamer/Data/Express/000/293/910/run293910_ls0328_streamExpress_StorageManager.dat',
'file:/eos/cms/store/t0streamer/Data/Express/000/293/910/run293910_ls0329_streamExpress_StorageManager.dat',
'file:/eos/cms/store/t0streamer/Data/Express/000/293/910/run293910_ls0330_streamExpress_StorageManager.dat',
'file:/eos/cms/store/t0streamer/Data/Express/000/293/910/run293910_ls0331_streamExpress_StorageManager.dat',
'file:/eos/cms/store/t0streamer/Data/Express/000/293/910/run293910_ls0332_streamExpress_StorageManager.dat',
'file:/eos/cms/store/t0streamer/Data/Express/000/293/910/run293910_ls0333_streamExpress_StorageManager.dat',
'file:/eos/cms/store/t0streamer/Data/Express/000/293/910/run293910_ls0334_streamExpress_StorageManager.dat',
'file:/eos/cms/store/t0streamer/Data/Express/000/293/910/run293910_ls0335_streamExpress_StorageManager.dat',
'file:/eos/cms/store/t0streamer/Data/Express/000/293/910/run293910_ls0336_streamExpress_StorageManager.dat',
'file:/eos/cms/store/t0streamer/Data/Express/000/293/910/run293910_ls0337_streamExpress_StorageManager.dat',
'file:/eos/cms/store/t0streamer/Data/Express/000/293/910/run293910_ls0338_streamExpress_StorageManager.dat',
'file:/eos/cms/store/t0streamer/Data/Express/000/293/910/run293910_ls0339_streamExpress_StorageManager.dat',
'file:/eos/cms/store/t0streamer/Data/Express/000/293/910/run293910_ls0340_streamExpress_StorageManager.dat',
'file:/eos/cms/store/t0streamer/Data/Express/000/293/910/run293910_ls0341_streamExpress_StorageManager.dat',
'file:/eos/cms/store/t0streamer/Data/Express/000/293/910/run293910_ls0342_streamExpress_StorageManager.dat',
'file:/eos/cms/store/t0streamer/Data/Express/000/293/910/run293910_ls0343_streamExpress_StorageManager.dat',
'file:/eos/cms/store/t0streamer/Data/Express/000/293/910/run293910_ls0344_streamExpress_StorageManager.dat',
'file:/eos/cms/store/t0streamer/Data/Express/000/293/910/run293910_ls0345_streamExpress_StorageManager.dat',
'file:/eos/cms/store/t0streamer/Data/Express/000/293/910/run293910_ls0346_streamExpress_StorageManager.dat',
'file:/eos/cms/store/t0streamer/Data/Express/000/293/910/run293910_ls0347_streamExpress_StorageManager.dat',
'file:/eos/cms/store/t0streamer/Data/Express/000/293/910/run293910_ls0348_streamExpress_StorageManager.dat',
'file:/eos/cms/store/t0streamer/Data/Express/000/293/910/run293910_ls0349_streamExpress_StorageManager.dat',
'file:/eos/cms/store/t0streamer/Data/Express/000/293/910/run293910_ls0350_streamExpress_StorageManager.dat',
'file:/eos/cms/store/t0streamer/Data/Express/000/293/910/run293910_ls0351_streamExpress_StorageManager.dat',
'file:/eos/cms/store/t0streamer/Data/Express/000/293/910/run293910_ls0352_streamExpress_StorageManager.dat',
'file:/eos/cms/store/t0streamer/Data/Express/000/293/910/run293910_ls0353_streamExpress_StorageManager.dat',
'file:/eos/cms/store/t0streamer/Data/Express/000/293/910/run293910_ls0354_streamExpress_StorageManager.dat',
'file:/eos/cms/store/t0streamer/Data/Express/000/293/910/run293910_ls0355_streamExpress_StorageManager.dat',
'file:/eos/cms/store/t0streamer/Data/Express/000/293/910/run293910_ls0356_streamExpress_StorageManager.dat',
'file:/eos/cms/store/t0streamer/Data/Express/000/293/910/run293910_ls0357_streamExpress_StorageManager.dat',
'file:/eos/cms/store/t0streamer/Data/Express/000/293/910/run293910_ls0358_streamExpress_StorageManager.dat',
'file:/eos/cms/store/t0streamer/Data/Express/000/293/910/run293910_ls0359_streamExpress_StorageManager.dat'


),
    inputFileTransitionsEachEvent = cms.untracked.bool(True)
    #firstEvent = cms.untracked.uint64(10123456835)
)



from CondCore.CondDB.CondDB_cfi import *
process.PoolDBESSource = cms.ESSource("PoolDBESSource",
        CondDB.clone(
connect = cms.string('sqlite_file:/eos/cms/store/group/dpg_ctpps/comm_ctpps/ctppspixnew3.db')),
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('CTPPSPixelGainCalibrationsRcd'),
        tag = cms.string("CTPPSPixelGainCalibNew_v3")
    ),
      cms.PSet(
        record = cms.string('CTPPSPixelDAQMappingRcd'),
        tag = cms.string("PixelDAQMapping_v1"),
        label = cms.untracked.string("RPix")
      ),
      cms.PSet(
        record = cms.string('CTPPSPixelAnalysisMaskRcd'),
        tag = cms.string("PixelAnalysisMask_v1"),
        label = cms.untracked.string("RPix")
      )


)
)


# raw-to-digi conversion
process.load("EventFilter.CTPPSRawToDigi.ctppsRawToDigi_cff")


############
process.o1 = cms.OutputModule("PoolOutputModule",
        outputCommands = cms.untracked.vstring('keep *'),
        fileName = cms.untracked.string('simevent_CTPPS_CLU_DB_real_2.root')
        )

process.load("RecoCTPPS.Configuration.recoCTPPS_cff")


process.ALL = cms.Path(process.ctppsRawToDigi*process.recoCTPPS)


process.outpath = cms.EndPath(process.o1)

process.schedule = cms.Schedule(process.ALL,process.outpath)

# filter all path with the production filter sequence
for path in process.paths:
  #  getattr(process,path)._seq = process.ProductionFilterSequence * getattr(process,path)._seq
    getattr(process,path)._seq =  getattr(process,path)._seq


