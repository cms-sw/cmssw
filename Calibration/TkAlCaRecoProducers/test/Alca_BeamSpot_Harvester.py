import FWCore.ParameterSet.Config as cms

process = cms.Process('HARVESTING')

###################################################################
# Input source
###################################################################
process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(
                                'file:AlcaBeamSpot.root'
                            ),
                            processingMode = cms.untracked.string('RunsAndLumis'))

###################################################################
# initialize MessageLogger
###################################################################
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr = cms.untracked.PSet(enable = cms.untracked.bool(False))
process.MessageLogger.cout = cms.untracked.PSet(
    threshold = cms.untracked.string('INFO'),
    default = cms.untracked.PSet(
        limit = cms.untracked.int32(0)
    ),
    AlcaBeamSpotHarvester = cms.untracked.PSet(
        reportEvery = cms.untracked.int32(100),
	limit = cms.untracked.int32(0)
    ),
    AlcaBeamSpotManager = cms.untracked.PSet(
        reportEvery = cms.untracked.int32(100),
	limit = cms.untracked.int32(0)
    )
)
process.MessageLogger.cout.enableStatistics = cms.untracked.bool(True)

###################################################################
# configure the harvester
###################################################################
process.load("Calibration.TkAlCaRecoProducers.AlcaBeamSpotHarvester_cff")
process.alcaBeamSpotHarvester.AlcaBeamSpotHarvesterParameters.BeamSpotOutputBase = cms.untracked.string("lumibased") #lumibase

###################################################################
# import of standard configurations
###################################################################
process.load('Configuration.StandardSequences.Services_cff')
process.load('Configuration.Geometry.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.EDMtoMEAtRunEnd_cff')
process.load('Configuration.StandardSequences.Harvesting_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run3_data', '')

process.configurationMetadata = cms.untracked.PSet(
    annotation = cms.untracked.string('step3 nevts:1'),
    name = cms.untracked.string('PyReleaseValidation')
)
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

#process.maxLuminosityBlocks=cms.untracked.PSet(
#         input=cms.untracked.int32(1)
#)

process.options = cms.untracked.PSet(
    Rethrow = cms.untracked.vstring('ProductNotFound'),
    fileMode = cms.untracked.string('FULLMERGE')
)

###################################################################
# Conditions Database
###################################################################
process.load("CondCore.CondDB.CondDB_cfi")
process.CondDB.connect = "sqlite_file:testbs.db"
process.PoolDBOutputService = cms.Service("PoolDBOutputService",
                                          process.CondDB,
                                          toPut = cms.VPSet(cms.PSet(
                                              record = cms.string('BeamSpotObjectsRcd'),
                                              tag = cms.string('TestLSBasedBS') )),
                                          loadBlobStreamer = cms.untracked.bool(False),
                                          timetype   = cms.untracked.string('lumiid')
                                          #    timetype   = cms.untracked.string('runnumber')
                                          )

process.alcaHarvesting = cms.Path(process.alcaBeamSpotHarvester)

# Schedule definition
process.schedule = cms.Schedule(process.alcaHarvesting)
