import FWCore.ParameterSet.Config as cms
process = cms.Process("CTPPS")

from Configuration.StandardSequences.Eras import eras

process = cms.Process('RECODQM')

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_hlt_relval', '')

# raw data source
#process.source = cms.Source("NewEventStreamFileReader",
#    fileNames = cms.untracked.vstring(
#        '/store/t0streamer/Data/Physics/000/286/591/run286591_ls0521_streamPhysics_StorageManager.dat',
#    )
#)
process.source = cms.Source('NewEventStreamFileReader',
    fileNames = cms.untracked.vstring(
'/store/t0streamer/Minidaq/A/000/303/982/run303982_ls0001_streamA_StorageManager.dat',
),
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

# raw-to-digi conversion
process.load("EventFilter.CTPPSRawToDigi.ctppsRawToDigi_cff")

# local RP reconstruction chain with standard settings
process.load("RecoCTPPS.Configuration.recoCTPPS_cff")

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
    process.ctppsDiamondLocalReconstruction
)

process.outpath = cms.EndPath(process.output) 
