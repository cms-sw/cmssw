import FWCore.ParameterSet.Config as cms
process = cms.Process("CTPPS")

# minimum of logs
#process.MessageLogger = cms.Service("MessageLogger",
#    statistics = cms.untracked.vstring(),
#    destinations = cms.untracked.vstring('cerr'),
#    cerr = cms.untracked.PSet(
#        threshold = cms.untracked.string('WARNING')
#    )
#)

# raw data source
#process.source = cms.Source("NewEventStreamFileReader",
#    fileNames = cms.untracked.vstring(
#        '/store/t0streamer/Data/Physics/000/286/591/run286591_ls0521_streamPhysics_StorageManager.dat',
#    )
#)
process.source = cms.Source('PoolSource',
    fileNames = cms.untracked.vstring(
        'root://eoscms.cern.ch:1094//eos/totem/data/ctpps/run284036.root',
    ),
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1000)
)

# diamonds mapping
process.totemDAQMappingESSourceXML_TimingDiamond = cms.ESSource("TotemDAQMappingESSourceXML",
  verbosity = cms.untracked.uint32(0),
  subSystem = cms.untracked.string("TimingDiamond"),
  configuration = cms.VPSet(
    # before diamonds inserted in DAQ
    cms.PSet(
      validityRange = cms.EventRange("1:min - 283819:max"),
      mappingFileNames = cms.vstring(),
      maskFileNames = cms.vstring()
    ),
    # after diamonds inserted in DAQ
    cms.PSet(
      validityRange = cms.EventRange("283820:min - 999999999:max"),
      mappingFileNames = cms.vstring("CondFormats/CTPPSReadoutObjects/xml/mapping_timing_diamond.xml"),
      maskFileNames = cms.vstring()
    )
  )
)

# raw-to-digi conversion
process.load('EventFilter.CTPPSRawToDigi.ctppsDiamondRawToDigi_cfi')
process.ctppsDiamondRawToDigi.rawDataTag = cms.InputTag("rawDataCollector")

# rechits production
process.load('Geometry.VeryForwardGeometry.geometryRP_cfi')
process.load('RecoCTPPS.TotemRPLocal.ctppsDiamondRecHits_cfi')

# local tracks fitter
process.load('RecoCTPPS.TotemRPLocal.ctppsDiamondLocalTracks_cfi')
#process.ctppsDiamondLocalTracks.trackingAlgorithmParams.threshold = cms.double(1.5)
#process.ctppsDiamondLocalTracks.trackingAlgorithmParams.sigma = cms.double(0)
#process.ctppsDiamondLocalTracks.trackingAlgorithmParams.resolution = cms.double(0.025) # in mm
#process.ctppsDiamondLocalTracks.trackingAlgorithmParams.pixel_efficiency_function = cms.string("(TMath::Erf((x-[0]+0.5*[1])/([2]/4)+2)+1)*TMath::Erfc((x-[0]-0.5*[1])/([2]/4)-2)/4")

process.output = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string("file:AOD.root"),
    outputCommands = cms.untracked.vstring(
        'drop *',
        'keep *_ctpps*_*_*',
    ),
)

# execution configuration
process.p = cms.Path(
    process.ctppsDiamondRawToDigi *
    process.ctppsDiamondRecHits *
    process.ctppsDiamondLocalTracks
)

process.outpath = cms.EndPath(process.output) 
