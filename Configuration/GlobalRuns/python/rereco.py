# Auto generated configuration file
# using: 
# Revision: 1.149 
# Source: /cvs_server/repositories/CMSSW/CMSSW/Configuration/PyReleaseValidation/python/ConfigBuilder.py,v 
# with command line options: promptCollisionReco -s RAW2DIGI,L1Reco,RECO,DQM,ALCA:SiStripCalZeroBias --datatier RECO --eventcontent RECO --conditions CRAFT09_R_V4::All --scenario pp --no_exec --data --magField AutoFromDBCurrent -n 100
import FWCore.ParameterSet.Config as cms

process = cms.Process('RECO')

# import of standard configurations
process.load('Configuration/StandardSequences/Services_cff')
process.load('FWCore/MessageService/MessageLogger_cfi')
process.load('Configuration/StandardSequences/GeometryExtended_cff')
process.load('Configuration/StandardSequences/MagneticField_AutoFromDBCurrent_cff')
process.load('Configuration/StandardSequences/RawToDigi_Data_cff')
process.load('Configuration/StandardSequences/L1Reco_cff')
process.load('Configuration/StandardSequences/Reconstruction_cff')
process.load('DQMOffline/Configuration/DQMOffline_cff')
process.load('Configuration/StandardSequences/AlCaRecoStreams_cff')
process.load('Configuration/StandardSequences/EndOfProcess_cff')
process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
process.load('Configuration/EventContent/EventContent_cff')

process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.10 $'),
    annotation = cms.untracked.string('promptCollisionReco nevts:100'),
    name = cms.untracked.string('PyReleaseValidation')
)
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)
process.options = cms.untracked.PSet(
    Rethrow = cms.untracked.vstring('ProductNotFound'),
    wantSummary = cms.untracked.bool(True) 
)
# Input source
process.source = cms.Source("PoolSource",
#    skipEvents = cms.untracked.uint32(278), 
    fileNames = cms.untracked.vstring(
"/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/122/314/1653DF8D-61D8-DE11-A283-00304879FBB2.root",
"/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/122/314/4259728C-61D8-DE11-BE08-000423D6A6F4.root",
"/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/122/314/04E779D1-60D8-DE11-9811-001D09F24498.root",
"/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/122/314/C4F34BCE-60D8-DE11-BA15-0019B9F581C9.root",
"/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/122/314/F6E6E5CD-60D8-DE11-A93B-0019B9F705A3.root",
"/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/122/314/2E6A82EE-60D8-DE11-B655-001D09F2423B.root",
"/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/122/314/CECE0A1C-60D8-DE11-8B50-001D09F26C5C.root",
"/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/122/314/2288321E-60D8-DE11-B57D-001D09F2AF1E.root"
#    fileNames = cms.untracked.vstring('/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/122/270/747C6137-F0D7-DE11-BE6C-001D09F242EF.root')
)
)
# Output definition
process.FEVT = cms.OutputModule("PoolOutputModule",
    splitLevel = cms.untracked.int32(0),
    outputCommands = process.RECOEventContent.outputCommands,
    fileName = cms.untracked.string('rerecoOutput.root'),
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('RECO'),
        filterName = cms.untracked.string('')
    )
)

# Other statements
process.GlobalTag.globaltag = 'FIRSTCOLL::All'


process.fifthCkfTrajectoryFilter.filterPset.minimumNumberOfHits = 2
process.fifthCkfTrajectoryFilter.filterPset.maxLostHits = 4
process.fifthCkfTrajectoryFilter.filterPset.maxConsecLostHits = 2
process.fifthCkfInOutTrajectoryFilter.filterPset.minimumNumberOfHits = 2
process.fifthCkfInOutTrajectoryFilter.filterPset.maxLostHits = 4
process.fifthCkfInOutTrajectoryFilter.filterPset.maxConsecLostHits = 2
process.fifthCkfTrajectoryBuilder.minNrOfHitsForRebuild = 2
process.fifthRKTrajectorySmoother.minHits = 2
process.fifthRKTrajectoryFitter.minHits = 2
process.fifthFittingSmootherWithOutlierRejection.MinNumberOfHits = 2
process.tobtecStepLoose.minNumberLayers = 2
process.tobtecStepLoose.maxNumberLostLayers = 2
process.tobtecStepLoose.dz_par1 = cms.vdouble(10.5, 4.0)
process.tobtecStepLoose.dz_par2 = cms.vdouble(10.5, 4.0)
process.tobtecStepLoose.d0_par1 = cms.vdouble(10.5, 4.0)
process.tobtecStepLoose.d0_par2 = cms.vdouble(10.5, 4.0)
process.tobtecStepLoose.chi2n_par = cms.double(100.0)
process.fifthSeeds.RegionFactoryPSet.RegionPSet.originHalfLength = 100
process.fifthSeeds.RegionFactoryPSet.RegionPSet.originRadius     = 10
process.Chi2MeasurementEstimator.MaxChi2 = 100


# Path and EndPath definitions
process.raw2digi_step = cms.Path(process.RawToDigi)
process.L1Reco_step = cms.Path(process.L1Reco)
process.reconstruction_step = cms.Path(process.reconstruction_withPixellessTk)
process.dqmoffline_step = cms.Path(process.DQMOffline)
process.endjob_step = cms.Path(process.endOfProcess)
process.out_step = cms.EndPath(process.FEVT)
process.ALCARECOStreamCombinedOutPath = cms.EndPath(process.ALCARECOStreamCombined)

# Schedule definition
process.schedule = cms.Schedule(process.raw2digi_step,process.L1Reco_step,process.reconstruction_step,process.dqmoffline_step,process.endjob_step,process.out_step)
