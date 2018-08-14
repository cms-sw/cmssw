import FWCore.ParameterSet.Config as cms

process = cms.Process("CTPPSRawToDigiTestPixelsOnly")

# minimum of logs
process.MessageLogger = cms.Service("MessageLogger",
    statistics = cms.untracked.vstring(),
    destinations = cms.untracked.vstring('cerr'),
    cerr = cms.untracked.PSet( threshold = cms.untracked.string('DEBUG') )
)

# raw data source
process.source = cms.Source("PoolSource",
  fileNames = cms.untracked.vstring(
    #'file:/afs/cern.ch/user/j/jkaspar/public/run273062_ls0001-2_stream.root'
   # '/store/express/Run2016H/ExpressPhysics/FEVT/Express-v2/000/283/877/00000/4EE44B0E-2499-E611-A155-02163E011938.root'
#'file:/afs/cern.ch/work/k/kas/public/PXtrees/PixelAlive_1294_151_RAW_v2.root'
#'root://eoscms//eos/cms/store/user/jjhollar/012017_PixelDAQTests/PixelAlive_1294_151_RAW_v2.root'
'root://eoscms//eos/cms/store/user/jjhollar/012017_PixelDAQTests/PixelAlive_1462_2_RAW.root',
'root://eoscms//eos/cms/store/user/jjhollar/012017_PixelDAQTests/PixelAlive_1463_2_RAW.root'
  ),
labelRawDataLikeMC = cms.untracked.bool(False), # for testing H8 data
duplicateCheckMode = cms.untracked.string("checkEachFile")
)

process.maxEvents = cms.untracked.PSet(
  input = cms.untracked.int32(-1)
)

# raw-to-digi conversion
process.load("EventFilter.CTPPSRawToDigi.ctppsRawToDigi_cff")
 
process.ctppsPixelDAQMappingESSourceXML.configuration = cms.VPSet(
    # example configuration block:
    cms.PSet(
        validityRange = cms.EventRange("1:min - 999999999:max"),
        mappingFileNames = cms.vstring("CondFormats/CTPPSReadoutObjects/xml/rpix_mapping_220_far.xml"),
        maskFileNames = cms.vstring("CondFormats/CTPPSReadoutObjects/xml/rpix_channel_mask_220_far.xml")
        )

    )

process.p = cms.Path(
  process.ctppsPixelDigis
)

# output configuration
process.output = cms.OutputModule("PoolOutputModule",
  fileName = cms.untracked.string("file:./ctpps_pixel_digi.root"),
  outputCommands = cms.untracked.vstring(
    'keep *'
 )
)

process.outpath = cms.EndPath(process.output)
