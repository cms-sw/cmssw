# Auto generated configuration file
# using: 
# Revision: 1.163 
# Source: /cvs_server/repositories/CMSSW/CMSSW/Configuration/PyReleaseValidation/python/ConfigBuilder.py,v 
# with command line options: rawreco_cfi -s RAW2DIGI,RECO --conditions=START3X_V21::All --eventcontent=FEVTSIM --mc --no_exec
import FWCore.ParameterSet.Config as cms

process = cms.Process('FILTER')

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load("Configuration.StandardSequences.Geometry_cff")
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')

process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.2 $'),
    annotation = cms.untracked.string('rawreco_cfi nevts:1'),
    name = cms.untracked.string('PyReleaseValidation')
)
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1000)
)
process.options = cms.untracked.PSet(

)
# Input source
process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(
'/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/123/151/6ADC6A1B-01DE-DE11-8FBD-00304879FA4A.root',
'/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/123/151/6ADC6A1B-01DE-DE11-8FBD-00304879FA4A.root'),
                            secondaryFileNames = cms.untracked.vstring(
'/store/data/BeamCommissioning09/MinimumBias/RAW/v1/000/123/151/3CE3F1C6-FADD-DE11-8AEA-001D09F251D1.root',
'/store/data/BeamCommissioning09/MinimumBias/RAW/v1/000/123/151/6C8F0233-FCDD-DE11-BF8E-001D09F297EF.root')
)

process.source.inputCommands = cms.untracked.vstring("keep *", "drop *_MEtoEDMConverter_*_*", "drop L1GlobalTriggerObjectMapRecord_hltL1GtObjectMap__HLT")

# Output definition
process.output = cms.OutputModule("PoolOutputModule",
    splitLevel = cms.untracked.int32(0),
    #outputCommands = process.FEVTSIMEventContent.outputCommands,
    outputCommands = cms.untracked.vstring("keep *"),
    fileName = cms.untracked.string('rechitskim.root'),
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string(''),
        filterName = cms.untracked.string('')
    ),
    SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring('filter_step'))
)

# Additional output definition

# The filter itself
process.load("DPGAnalysis.Skims.filterRecHits_cfi")

# Path and EndPath definitions
process.filter_step = cms.Path(process.recHitEnergyFilter)
process.out_step = cms.EndPath(process.output)

# Schedule definition
process.schedule = cms.Schedule(process.filter_step, process.out_step)
