import FWCore.ParameterSet.Config as cms

process = cms.Process("DiamondSampic")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(10000) )

process.source = cms.Source("EmptySource")

process.load('RecoPPS.Local.totemTimingLocalReconstruction_cff')

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_data', '')

################
#digi converter
################
process.totemTimingRawToDigi = cms.EDProducer('DiamondSampicDigiProducer',
	#input path of the testbeam data
	sampicFilesVec=cms.vstring("/eos/cms/store/group/dpg_ctpps/comm_ctpps/201905_DesyTestbeam/MergedDev/Ntuple_runsampic_159_runtelescope_636.root"),
	################
	#channel mapping
	################
	idsMapping = cms.VPSet(
		cms.PSet(detId = cms.vuint32(2054160384,2054553600,2056257536,2056650752), treeChId = cms.uint32(8)),
		cms.PSet(detId = cms.vuint32(2054164480,2054557696,2056261632,2056654848), treeChId = cms.uint32(9)),
		cms.PSet(detId = cms.vuint32(2054168576,2054561792,2056265728,2056658944), treeChId = cms.uint32(10)),
		cms.PSet(detId = cms.vuint32(2054172672,2054565888,2056269824,2056663040), treeChId = cms.uint32(11)),
		cms.PSet(detId = cms.vuint32(2054176768,2054569984,2056273920,2056667136), treeChId = cms.uint32(12)),
		cms.PSet(detId = cms.vuint32(2054180864,2054574080,2056278016,2056671232), treeChId = cms.uint32(13)),
		cms.PSet(detId = cms.vuint32(2054184960,2054578176,2056282112,2056675328), treeChId = cms.uint32(14)),
		cms.PSet(detId = cms.vuint32(2054189056,2054582272,2056286208,2056679424), treeChId = cms.uint32(15)),
		cms.PSet(detId = cms.vuint32(2054193152,2054586368,2056290304,2056683520), treeChId = cms.uint32(16)),
		cms.PSet(detId = cms.vuint32(2054197248,2054590464,2056294400,2056687616), treeChId = cms.uint32(17)),
		cms.PSet(detId = cms.vuint32(2054201344,2054594560,2056298496,2056691712), treeChId = cms.uint32(18)),
		cms.PSet(detId = cms.vuint32(2054205440,2054598656,2056302592,2056695808), treeChId = cms.uint32(19)),

		cms.PSet(detId = cms.vuint32(2054291456,2054422528,2056388608,2056519680), treeChId = cms.uint32(20)),
		cms.PSet(detId = cms.vuint32(2054295552,2054426624,2056392704,2056523776), treeChId = cms.uint32(21)),
		cms.PSet(detId = cms.vuint32(2054299648,2054430720,2056396800,2056527872), treeChId = cms.uint32(22)),
		cms.PSet(detId = cms.vuint32(2054303744,2054434816,2056400896,2056531968), treeChId = cms.uint32(23)),
		cms.PSet(detId = cms.vuint32(2054307840,2054438912,2056404992,2056536064), treeChId = cms.uint32(24)),
		cms.PSet(detId = cms.vuint32(2054311936,2054443008,2056409088,2056540160), treeChId = cms.uint32(25)),
		cms.PSet(detId = cms.vuint32(2054316032,2054447104,2056413184,2056544256), treeChId = cms.uint32(26)),
		cms.PSet(detId = cms.vuint32(2054320128,2054451200,2056417280,2056548352), treeChId = cms.uint32(27)),
		cms.PSet(detId = cms.vuint32(2054324224,2054455296,2056421376,2056552448), treeChId = cms.uint32(28)),
		cms.PSet(detId = cms.vuint32(2054328320,2054459392,2056425472,2056556544), treeChId = cms.uint32(29)),
		cms.PSet(detId = cms.vuint32(2054332416,2054463488,2056429568,2056560640), treeChId = cms.uint32(30)),
		cms.PSet(detId = cms.vuint32(2054336512,2054467584,2056433664,2056564736), treeChId = cms.uint32(31))

	)
)

################
#geometry
################
process.load('Geometry.VeryForwardGeometry.geometryRPFromDD_2021_cfi')

################
#calib
################
 
#load calibrations from json    
#process.totemTimingRecHits.timingCalibrationTag= cms.string('ppsTimingCalibrationESSource:TotemTimingCalibration')
#process.ppsTimingCalibrationESSource = cms.ESSource('PPSTimingCalibrationESSource',
#  calibrationFile = cms.FileInPath('RecoPPS/Local/data/timing_offsets_ufsd_2018.dec18.cal.json'),#calibration file does not yet exist in db
#  subDetector = cms.uint32(1),
#  appendToDataLabel = cms.string('TotemTimingCalibration')
#)



process.totemTimingRecHits.mergeTimePeaks= cms.bool(False)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('file:diamondSampicReco.root')
    
)

process.p = cms.Path(process.totemTimingRawToDigi*
	process.diamondSampicLocalReconstruction
)

process.outpath = cms.EndPath(process.out)


