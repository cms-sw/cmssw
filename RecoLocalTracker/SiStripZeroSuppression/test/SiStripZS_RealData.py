# Auto generated configuration file
# using: 
# Revision: 1.232.2.6.2.2 
# Source: /cvs_server/repositories/CMSSW/CMSSW/Configuration/PyReleaseValidation/python/ConfigBuilder.py,v 
# with command line options: hiReco -n 10 --scenario HeavyIons -s RAW2DIGI,L1Reco,RECO --processName RECO --data --datatier RECO --eventcontent FEVTDEBUG --geometry DB --filein /store/hidata/HIRun2010/HIAllPhysics/RAW/v1/000/152/698/FEA10166-D9FB-DF11-A90C-0019B9F72F97.root --fileout hiReco_RECO.root --conditions FrontierConditions_GlobalTag,GR_R_39X_V1::All --no_exec
import FWCore.ParameterSet.Config as cms

process = cms.Process('RECO')

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.StandardSequences.GeometryDB_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_cff')
process.load('Configuration.StandardSequences.RawToDigi_Data_cff')
process.load('Configuration.StandardSequences.L1Reco_cff')
process.load('Configuration.StandardSequences.ReconstructionHeavyIons_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load('Configuration.EventContent.EventContentHeavyIons_cff')

process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.232.2.6.2.2 $'),
    annotation = cms.untracked.string('hiReco nevts:10'),
    name = cms.untracked.string('PyReleaseValidation')
)
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)

# Input source
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:/store/hidata/HIRun2010/HIAllPhysics/RAW/v1/000/152/698/FEA10166-D9FB-DF11-A90C-0019B9F72F97.root')
)

process.options = cms.untracked.PSet(

)

# Output definition

process.FEVTDEBUGoutput = cms.OutputModule("PoolOutputModule",
    splitLevel = cms.untracked.int32(0),
    outputCommands = process.FEVTDEBUGEventContent.outputCommands,
    fileName = cms.untracked.string('hiReco_RECO.root'),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string(''),
        dataTier = cms.untracked.string('RECO')
    )
)

# Additional output definition

# Other statements
process.GlobalTag.globaltag = 'GR_R_39X_V1::All'

# Path and EndPath definitions
process.raw2digi_step = cms.Path(process.RawToDigi)

process.L1Reco_step = cms.Path(process.L1Reco)

process.reconstruction_step = cms.Path(process.reconstructionHeavyIons)

process.endjob_step = cms.EndPath(process.endOfProcess)

process.FEVTDEBUGoutput_step = cms.EndPath(process.FEVTDEBUGoutput)


# Schedule definition
process.schedule = cms.Schedule(process.raw2digi_step,process.L1Reco_step,process.reconstruction_step,process.endjob_step,process.FEVTDEBUGoutput_step)
