# Auto generated configuration file
# using: 
# Revision: 1.123 
# Source: /cvs_server/repositories/CMSSW/CMSSW/Configuration/PyReleaseValidation/python/ConfigBuilder.py,v 
# with command line options: ZMM.py -s DIGI2RAW,RAW2DIGI --no_exec --conditions=FrontierConditions_GlobalTag,MC_31X_V3::All
import FWCore.ParameterSet.Config as cms

process = cms.Process('RAW2DIGI')

# import of standard configurations
process.load('Configuration/StandardSequences/Services_cff')
process.load('FWCore/MessageService/MessageLogger_cfi')
process.load('Configuration/StandardSequences/MixingNoPileUp_cff')
process.load('Configuration/StandardSequences/GeometryIdeal_cff')
process.load('Configuration/StandardSequences/MagneticField_38T_cff')
process.load('Configuration/StandardSequences/DigiToRaw_cff')
process.load('Configuration/StandardSequences/RawToDigi_cff')
process.load('Configuration/StandardSequences/EndOfProcess_cff')
process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
process.load('Configuration/EventContent/EventContent_cff')
process.load('EventFilter/CSCRawToDigi/cscdigivalidator_cfi')

process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.1 $'),
    annotation = cms.untracked.string('ZMM.py nevts:1'),
    name = cms.untracked.string('PyReleaseValidation')
)
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.options = cms.untracked.PSet(
    Rethrow = cms.untracked.vstring('ProductNotFound')
)
# Input source
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:ZMM_cfi_GEN_SIM_DIGI_L1.root')
)

# Additional output definition
process.ana = cms.EDAnalyzer('EventContentAnalyzer')

# Other statements
process.GlobalTag.globaltag = 'MC_31X_V3::All'

# Path and EndPath definitions
process.digi2raw_step = cms.Path(process.DigiToRaw)
process.raw2digi_step = cms.Path(process.RawToDigi)
process.endjob_step = cms.Path(process.cscDigiValidator+process.endOfProcess)
#process.out_step = cms.EndPath(process.output)

# Schedule definition
process.schedule = cms.Schedule(process.digi2raw_step,process.raw2digi_step,process.endjob_step)#,process.out_step)
