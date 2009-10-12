# Auto generated configuration file
# using: 
# Revision: 1.123 
# Source: /cvs_server/repositories/CMSSW/CMSSW/Configuration/PyReleaseValidation/python/ConfigBuilder.py,v 
# with command line options: alCaRecoSplitting -s ALCA:SiStripCalZeroBias+DQM --datatier RECO --eventcontent RECO --conditions FrontierConditions_GlobalTag,GR09_31X_V4P::All -n -1 --no_exec
import FWCore.ParameterSet.Config as cms

process = cms.Process('ALCA')

# import of standard configurations
process.load('Configuration/StandardSequences/Services_cff')
process.load('FWCore/MessageService/MessageLogger_cfi')

process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.1 $'),
    annotation = cms.untracked.string('step3_RELVAL nevts:-1'),
    name = cms.untracked.string('PyReleaseValidation')
)
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.options = cms.untracked.PSet(
    Rethrow = cms.untracked.vstring('ProductNotFound'),
    wantSummary = cms.untracked.bool(True)
)
# Input source
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:ALCACombined.root')
)

# Additional output definition
process.ALCARECOStreamSiStripCalZeroBias = cms.OutputModule("PoolOutputModule",
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOSiStripCalZeroBias:RECO')
    ),
    outputCommands = cms.untracked.vstring('drop *', 
        'keep *_ALCARECOSiStripCalZeroBias_*_*', 
        'keep *_calZeroBiasClusters_*_*', 
        'keep *_MEtoEDMConverter_*_*'),
    fileName = cms.untracked.string('SiStripCalZeroBias.root'),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('SiStripCalZeroBias'),
        dataTier = cms.untracked.string('ALCARECO')
    )
)

# Path and EndPath definitions
process.ALCARECOStreamSiStripCalZeroBiasOutPath = cms.EndPath(process.ALCARECOStreamSiStripCalZeroBias)

# Schedule definition
process.schedule = cms.Schedule(process.ALCARECOStreamSiStripCalZeroBiasOutPath)
