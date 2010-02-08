# Auto generated configuration file
# using: 
# Revision: 1.151 
# Source: /cvs_server/repositories/CMSSW/CMSSW/Configuration/PyReleaseValidation/python/ConfigBuilder.py,v 
# with command line options: alCaRecoSplitting -s ALCA:MuAlCalIsolatedMu+TkAlCosmics0T+MuAlStandAloneCosmics+MuAlGlobalCosmics+HcalCalHOCosmics+TkAlBeamHalo+MuAlBeamHaloOverlaps+MuAlBeamHalo --datatier ALCARECO --eventcontent ALCARECO --conditions FrontierConditions_GlobalTag,GR09_P_V6::All -n -1 --no_exec
import FWCore.ParameterSet.Config as cms

process = cms.Process('ALCA')

# import of standard configurations
process.load('Configuration/StandardSequences/Services_cff')
process.load('FWCore/MessageService/MessageLogger_cfi')
process.load('Configuration/EventContent/AlCaRecoOutput_cff')

process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.1 $'),
    annotation = cms.untracked.string('alCaRecoSplitting nevts:-1'),
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
process.ALCARECOStreamMuAlStandAloneCosmics = cms.OutputModule("PoolOutputModule",
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOMuAlStandAloneCosmics:RECO')
    ),
    outputCommands = process.OutALCARECOMuAlStandAloneCosmics_noDrop.outputCommands,
    fileName = cms.untracked.string('MuAlStandAloneCosmics.root'),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('StreamMuAlStandAloneCosmics'),
        dataTier = cms.untracked.string('ALCARECO')
    )
)
process.ALCARECOStreamTkAlCosmics0T = cms.OutputModule("PoolOutputModule",
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOTkAlCosmicsCTF0T:RECO', 
            'pathALCARECOTkAlCosmicsCosmicTF0T:RECO', 
            'pathALCARECOTkAlCosmicsRS0T:RECO')
    ),
    outputCommands = process.OutALCARECOTkAlCosmics0T_noDrop.outputCommands,
    fileName = cms.untracked.string('TkAlCosmics0T.root'),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('StreamTkAlCosmics0T'),
        dataTier = cms.untracked.string('ALCARECO')
    )
)
process.ALCARECOStreamMuAlBeamHaloOverlaps = cms.OutputModule("PoolOutputModule",
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOMuAlBeamHaloOverlaps:RECO')
    ),
    outputCommands = process.OutALCARECOMuAlBeamHaloOverlaps_noDrop.outputCommands,
    fileName = cms.untracked.string('MuAlBeamHaloOverlaps.root'),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('StreamMuAlBeamHaloOverlaps'),
        dataTier = cms.untracked.string('ALCARECO')
    )
)
process.ALCARECOStreamHcalCalHOCosmics = cms.OutputModule("PoolOutputModule",
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOHcalCalHOCosmics:RECO')
    ),
    outputCommands = process.OutALCARECOHcalCalHOCosmics_noDrop.outputCommands,
    fileName = cms.untracked.string('HcalCalHOCosmics.root'),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('StreamHcalCalHOCosmics'),
        dataTier = cms.untracked.string('ALCARECO')
    )
)
process.ALCARECOStreamMuAlBeamHalo = cms.OutputModule("PoolOutputModule",
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOMuAlBeamHalo:RECO')
    ),
    outputCommands = process.OutALCARECOMuAlBeamHalo_noDrop.outputCommands,
    fileName = cms.untracked.string('MuAlBeamHalo.root'),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('StreamMuAlBeamHalo'),
        dataTier = cms.untracked.string('ALCARECO')
    )
)
process.ALCARECOStreamMuAlGlobalCosmics = cms.OutputModule("PoolOutputModule",
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOMuAlGlobalCosmics:RECO')
    ),
    outputCommands = process.OutALCARECOMuAlGlobalCosmics_noDrop.outputCommands,
    fileName = cms.untracked.string('MuAlGlobalCosmics.root'),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('StreamMuAlGlobalCosmics'),
        dataTier = cms.untracked.string('ALCARECO')
    )
)
process.ALCARECOStreamMuAlCalIsolatedMu = cms.OutputModule("PoolOutputModule",
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOMuAlCalIsolatedMu:RECO')
    ),
    outputCommands = process.OutALCARECOMuAlCalIsolatedMu_noDrop.outputCommands,
    fileName = cms.untracked.string('MuAlCalIsolatedMu.root'),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('StreamMuAlCalIsolatedMu'),
        dataTier = cms.untracked.string('ALCARECO')
    )
)
process.ALCARECOStreamTkAlBeamHalo = cms.OutputModule("PoolOutputModule",
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOTkAlBeamHalo:RECO')
    ),
    outputCommands = process.OutALCARECOTkAlBeamHalo_noDrop.outputCommands,
    fileName = cms.untracked.string('TkAlBeamHalo.root'),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('StreamTkAlBeamHalo'),
        dataTier = cms.untracked.string('ALCARECO')
    )
)

# Path and EndPath definitions
process.ALCARECOStreamMuAlStandAloneCosmicsOutPath = cms.EndPath(process.ALCARECOStreamMuAlStandAloneCosmics)
process.ALCARECOStreamTkAlCosmics0TOutPath = cms.EndPath(process.ALCARECOStreamTkAlCosmics0T)
process.ALCARECOStreamMuAlBeamHaloOverlapsOutPath = cms.EndPath(process.ALCARECOStreamMuAlBeamHaloOverlaps)
process.ALCARECOStreamHcalCalHOCosmicsOutPath = cms.EndPath(process.ALCARECOStreamHcalCalHOCosmics)
process.ALCARECOStreamMuAlBeamHaloOutPath = cms.EndPath(process.ALCARECOStreamMuAlBeamHalo)
process.ALCARECOStreamMuAlGlobalCosmicsOutPath = cms.EndPath(process.ALCARECOStreamMuAlGlobalCosmics)
process.ALCARECOStreamMuAlCalIsolatedMuOutPath = cms.EndPath(process.ALCARECOStreamMuAlCalIsolatedMu)
process.ALCARECOStreamTkAlBeamHaloOutPath = cms.EndPath(process.ALCARECOStreamTkAlBeamHalo)

# Schedule definition
process.schedule = cms.Schedule(process.ALCARECOStreamMuAlStandAloneCosmicsOutPath,process.ALCARECOStreamTkAlCosmics0TOutPath,process.ALCARECOStreamMuAlBeamHaloOverlapsOutPath,process.ALCARECOStreamHcalCalHOCosmicsOutPath,process.ALCARECOStreamMuAlBeamHaloOutPath,process.ALCARECOStreamMuAlGlobalCosmicsOutPath,process.ALCARECOStreamMuAlCalIsolatedMuOutPath,process.ALCARECOStreamTkAlBeamHaloOutPath)
