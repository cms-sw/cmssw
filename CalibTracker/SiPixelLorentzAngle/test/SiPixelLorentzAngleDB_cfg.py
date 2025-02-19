import FWCore.ParameterSet.Config as cms

process = cms.Process("LA")

process.load("Geometry.TrackerGeometryBuilder.trackerGeometry_cfi")

process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")

process.load("Geometry.TrackerSimData.trackerSimGeometryXML_cfi")

process.load("Configuration.StandardSequences.MagneticField_cff")

process.load("CondCore.DBCommon.CondDBCommon_cfi")

# process.load("Configuration.StandardSequences.FakeConditions_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
#process.GlobalTag.globaltag = 'IDEAL_V6::All'

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.source = cms.Source("EmptyIOVSource",
	timetype = cms.string("runnumber"),
	firstRun = cms.untracked.uint32(1),
	lastRun = cms.untracked.uint32(1),
	firstValue = cms.uint64(1),
	lastValue = cms.uint64(1),
	interval = cms.uint64(1)
)

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
  timetype = cms.untracked.string("runnumber"),
	connect = cms.string("sqlite_file:SiPixelLorentzAngle.db") ,
# 	connect = cms.string("oracle://cms_orcoff_int2r/CMS_COND_PIXEL") 
	toPut = cms.VPSet(
		cms.PSet(
			record = cms.string("SiPixelLorentzAngleRcd"),
			tag = cms.string("trivial_LorentzAngle")	
		)	
	),
	
	DBParameters = cms.PSet(
		authenticationPath = cms.untracked.string(""),
		messageLevel = cms.untracked.int32(2)	
	)				
)

process.lorentzAngle = cms.EDAnalyzer("SiPixelLorentzAngleDB",
	magneticField = cms.double(3.8),
	bPixLorentzAnglePerTesla = cms.double(0.106),
	fPixLorentzAnglePerTesla = cms.double(0.054),
# 	in case lorentz angle values for bpix should be read from file -> not implemented yet
	useFile = cms.bool(False),
	fileName = cms.string('lorentzFit.txt')					
)

process.p = cms.Path(process.lorentzAngle)
