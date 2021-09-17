# Auto generated configuration file
# using: 
# Revision: 1.172 
# Source: /cvs_server/repositories/CMSSW/CMSSW/Configuration/PyReleaseValidation/python/ConfigBuilder.py,v 
# with command line options: step3 -s HARVESTING:dqmHarvesting --conditions auto:com10 --scenario pp --no_exec --data
import FWCore.ParameterSet.Config as cms

process = cms.Process('HARVESTING')

# Input source
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
      'file:AlcaBeamSpot.root'
    ),
    processingMode = cms.untracked.string('RunsAndLumis')
)


# initialize MessageLogger
process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.MessageLogger.cerr = cms.untracked.PSet(enable = cms.untracked.bool(False))
process.MessageLogger.cout = cms.untracked.PSet(
    threshold = cms.untracked.string('INFO'),
    default = cms.untracked.PSet(
       limit = cms.untracked.int32(0)
    ),
    AlcaBeamSpotHarvester = cms.untracked.PSet(
        #reportEvery = cms.untracked.int32(100) # every 1000th only
	limit = cms.untracked.int32(0)
    ),
    AlcaBeamSpotManager = cms.untracked.PSet(
        #reportEvery = cms.untracked.int32(100) # every 1000th only
	limit = cms.untracked.int32(0)
    )
)
#process.MessageLogger.cout.enableStatistics = cms.untracked.bool(True)

process.load("Calibration.TkAlCaRecoProducers.AlcaBeamSpotHarvester_cff")

process.alcaBeamSpotHarvester.AlcaBeamSpotHarvesterParameters.BeamSpotOutputBase = cms.untracked.string("lumibased") #lumibase

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('Configuration.StandardSequences.GeometryExtended_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_cff')
process.load('Configuration.StandardSequences.EDMtoMEAtRunEnd_cff')
process.load('Configuration.StandardSequences.Harvesting_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load('Configuration.EventContent.EventContent_cff')

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

# Additional output definition

# Other statements
process.GlobalTag.globaltag = 'GR_R_38X_V9::All'

# Path and EndPath definitions
process.edmtome_step = cms.Path(process.EDMtoME)
process.validationpreprodHarvesting = cms.Path(process.postValidation*process.hltpostvalidation_preprod)
process.validationprodHarvesting = cms.Path(process.postValidation*process.hltpostvalidation_prod)
process.dqmHarvesting = cms.Path(process.DQMOffline_SecondStep*process.DQMOffline_Certification)
process.validationHarvesting = cms.Path(process.postValidation*process.hltpostvalidation)
process.validationHarvestingFS = cms.Path(process.HarvestingFastSim)
process.dqmHarvestingPOG = cms.Path(process.DQMOffline_SecondStep_PrePOG)
process.dqmsave_step = cms.Path(process.DQMSaver)




process.load("CondCore.DBCommon.CondDBCommon_cfi")


process.CondDBCommon.connect = "sqlite_file:testbs.db"
process.PoolDBOutputService = cms.Service("PoolDBOutputService",
                                          process.CondDBCommon,
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
