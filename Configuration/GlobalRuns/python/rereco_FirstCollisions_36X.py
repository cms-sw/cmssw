# Auto generated configuration file
# using: 
# Revision: 1.169 
# Source: /cvs_server/repositories/CMSSW/CMSSW/Configuration/PyReleaseValidation/python/ConfigBuilder.py,v 
# with command line options: reco_FirstCollisions_36X -s RAW2DIGI,L1Reco,RECO,DQM --data --magField AutoFromDBCurrent --scenario pp --datatier RECO --eventcontent RECO --conditions GR_R_36X_V6::All --customise Configuration/GlobalRuns/customise_Collision_36X.py --no_exec --python_filename=rereco_FirstCollisions_36X.py
import FWCore.ParameterSet.Config as cms

process = cms.Process('RECO')

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.StandardSequences.GeometryExtended_cff')
process.load('Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff')
process.load('Configuration.StandardSequences.RawToDigi_Data_cff')
process.load('Configuration.StandardSequences.L1Reco_cff')
process.load('Configuration.StandardSequences.Reconstruction_cff')
process.load('DQMOffline.Configuration.DQMOffline_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load('Configuration.EventContent.EventContent_cff')

process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.3 $'),
    annotation = cms.untracked.string('reco_FirstCollisions_36X nevts:1'),
    name = cms.untracked.string('PyReleaseValidation')
)
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.options = cms.untracked.PSet(

)
# Input source
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('reco_FirstCollisions_36X_DIGI2RAW.root'),
                            inputCommands = cms.untracked.vstring("keep *","drop *_*_*_RECO")
                            
)

# Output definition
process.output = cms.OutputModule("PoolOutputModule",
    splitLevel = cms.untracked.int32(0),
    outputCommands = process.RECOEventContent.outputCommands,
    fileName = cms.untracked.string('reco_FirstCollisions_36X_RAW2DIGI_L1Reco_RECO_DQM.root'),
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('RECO'),
        filterName = cms.untracked.string('')
    )
)

# Additional output definition

# Other statements
process.GlobalTag.globaltag = 'GR_R_36X_V6::All'

# Path and EndPath definitions
process.raw2digi_step = cms.Path(process.RawToDigi)
process.L1Reco_step = cms.Path(process.L1Reco)
process.reconstruction_step = cms.Path(process.reconstruction)
process.dqmoffline_step = cms.Path(process.DQMOffline)
process.endjob_step = cms.Path(process.endOfProcess)
process.out_step = cms.EndPath(process.output)

# Schedule definition
process.schedule = cms.Schedule(process.raw2digi_step,process.L1Reco_step,process.reconstruction_step,process.dqmoffline_step,process.endjob_step,process.out_step)


# Automatic addition of the customisation function

def customise(process):
    from Configuration.GlobalRuns.reco_TLR_36X import customisePPData
    process=customisePPData(process)
    return process






# End of customisation function definition

process = customise(process)
