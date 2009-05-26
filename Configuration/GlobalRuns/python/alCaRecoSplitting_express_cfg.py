# Auto generated configuration file
# using: 
# Revision: 1.112 
# Source: /cvs/CMSSW/CMSSW/Configuration/PyReleaseValidation/python/ConfigBuilder.py,v 
# with command line options: step3_RELVAL -s ALCA:MuAlCalIsolatedMu+RpcCalHLT+TkAlCosmicsHLT+TkAlCosmics0T+MuAlStandAloneCosmics+MuAlGlobalCosmics+HcalCalHOCosmics -n 1000 --filein file:reco.root --conditions FrontierConditions_GlobalTag,CRAFT_30X::All --no_exec --datatier ALCARECO --eventcontent RECO --scenario cosmics
import FWCore.ParameterSet.Config as cms

process = cms.Process('ALCA')

# import of standard configurations
process.load('Configuration/StandardSequences/Services_cff')
process.load('FWCore/MessageService/MessageLogger_cfi')
process.load('Configuration/StandardSequences/EndOfProcess_cff')

process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.112 $'),
    annotation = cms.untracked.string('step3_RELVAL nevts:1000'),
    name = cms.untracked.string('PyReleaseValidation')
)
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1000)
)
process.options = cms.untracked.PSet(
    Rethrow = cms.untracked.vstring('ProductNotFound')
)
# Input source
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:ALCACombined.root')
)

# Additional output definition
process.ALCARECOStreamMuAlStandAloneCosmics = cms.OutputModule("PoolOutputModule",
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOMuAlStandAloneCosmics:EXPRESS')
    ),
    outputCommands = cms.untracked.vstring('drop *', 
        'keep *_ALCARECOMuAlStandAloneCosmics_*_*', 
        'keep *_cosmicMuons_*_*', 
        'keep *_cosmictrackfinderP5_*_*', 
        'keep Si*Cluster*_*_*_*', 
        'keep *_muonCSCDigis_*_*', 
        'keep *_muonDTDigis_*_*', 
        'keep *_muonRPCDigis_*_*', 
        'keep *_dt1DRecHits_*_*', 
        'keep *_dt2DSegments_*_*', 
        'keep *_dt4DSegments_*_*', 
        'keep *_csc2DRecHits_*_*', 
        'keep *_cscSegments_*_*', 
        'keep *_rpcRecHits_*_*'),
    fileName = cms.untracked.string('ALCARECOMuAlStandAloneCosmics.root'),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('StreamALCARECOMuAlStandAloneCosmics'),
        dataTier = cms.untracked.string('ALCARECO')
    )
)
process.ALCARECOStreamHcalCalHOCosmics = cms.OutputModule("PoolOutputModule",
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOHcalCalHOCosmics:EXPRESS')
    ),
    outputCommands = cms.untracked.vstring('drop *', 
        'keep HOCalibVariabless_*_*_*'),
    fileName = cms.untracked.string('ALCARECOHcalCalHOCosmics.root'),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('StreamALCARECOHcalCalHOCosmics'),
        dataTier = cms.untracked.string('ALCARECO')
    )
)
process.ALCARECOStreamTkAlCosmics0T = cms.OutputModule("PoolOutputModule",
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOTkAlCosmicsCTF0T:EXPRESS', 
            'pathALCARECOTkAlCosmicsCosmicTF0T:EXPRESS', 
            'pathALCARECOTkAlCosmicsRS0T:EXPRESS')
    ),
    outputCommands = cms.untracked.vstring('drop *', 
        'keep *_ALCARECOTkAlCosmics*0T_*_*', 
        'keep *_eventAuxiliaryHistoryProducer_*_*', 
        'keep L1GlobalTriggerReadoutRecord_gtDigis_*_*', 
        'keep L1MuGMTReadoutCollection_gtDigis_*_*', 
        'keep Si*Cluster*_si*Clusters_*_*', 
        'keep *_MEtoEDMConverter_*_*'),
    fileName = cms.untracked.string('ALCARECOTkAlCosmics0T.root'),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('StreamALCARECOTkAlCosmics0T'),
        dataTier = cms.untracked.string('ALCARECO')
    )
)
process.ALCARECOStreamTkAlCosmicsHLT = cms.OutputModule("PoolOutputModule",
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOTkAlCosmicsCTFHLT:EXPRESS', 
            'pathALCARECOTkAlCosmicsCosmicTFHLT:EXPRESS', 
            'pathALCARECOTkAlCosmicsRSHLT:EXPRESS')
    ),
    outputCommands = cms.untracked.vstring('drop *', 
        'keep *_ALCARECOTkAlCosmicsCTF_*_*', 
        'keep *_ALCARECOTkAlCosmicsCosmicTF_*_*', 
        'keep *_ALCARECOTkAlCosmicsRS_*_*', 
        'keep *_eventAuxiliaryHistoryProducer_*_*', 
        'keep L1GlobalTriggerReadoutRecord_gtDigis_*_*', 
        'keep L1MuGMTReadoutCollection_gtDigis_*_*', 
        'keep Si*Cluster*_si*Clusters_*_*', 
        'keep *_MEtoEDMConverter_*_*'),
    fileName = cms.untracked.string('ALCARECOTkAlCosmicsHLT.root'),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('StreamALCARECOTkAlCosmicsHLT'),
        dataTier = cms.untracked.string('ALCARECO')
    )
)
process.ALCARECOStreamRpcCalHLT = cms.OutputModule("PoolOutputModule",
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECORpcCalHLT:EXPRESS')
    ),
    outputCommands = cms.untracked.vstring('drop *', 
        'keep *_muonDTDigis_*_*', 
        'keep CSCDetIdCSCWireDigiMuonDigiCollection_*_*_*', 
        'keep CSCDetIdCSCStripDigiMuonDigiCollection_*_*_*', 
        'keep DTLayerIdDTDigiMuonDigiCollection_*_*_*', 
        'keep *_dt4DSegments_*_*', 
        'keep *_cscSegments_*_*', 
        'keep RPCDetIdRPCDigiMuonDigiCollection_*_*_*', 
        'keep L1MuGMTCands_*_*_*', 
        'keep L1MuGMTReadoutCollection_*_*_*'),
    fileName = cms.untracked.string('ALCARECORpcCalHLT.root'),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('StreamALCARECORpcCalHLT'),
        dataTier = cms.untracked.string('ALCARECO')
    )
)
process.ALCARECOStreamMuAlGlobalCosmics = cms.OutputModule("PoolOutputModule",
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOMuAlGlobalCosmics:EXPRESS')
    ),
    outputCommands = cms.untracked.vstring('drop *', 
        'keep *_ALCARECOMuAlGlobalCosmics_*_*', 
        'keep *_cosmicMuons_*_*', 
        'keep *_cosmictrackfinderP5_*_*', 
        'keep Si*Cluster*_*_*_*', 
        'keep *_muonCSCDigis_*_*', 
        'keep *_muonDTDigis_*_*', 
        'keep *_muonRPCDigis_*_*', 
        'keep *_dt1DRecHits_*_*', 
        'keep *_dt2DSegments_*_*', 
        'keep *_dt4DSegments_*_*', 
        'keep *_csc2DRecHits_*_*', 
        'keep *_cscSegments_*_*', 
        'keep *_rpcRecHits_*_*'),
    fileName = cms.untracked.string('ALCARECOMuAlGlobalCosmics.root'),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('StreamALCARECOMuAlGlobalCosmics'),
        dataTier = cms.untracked.string('ALCARECO')
    )
)
process.ALCARECOStreamMuAlCalIsolatedMu = cms.OutputModule("PoolOutputModule",
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOMuAlCalIsolatedMu:EXPRESS')
    ),
    outputCommands = cms.untracked.vstring('drop *', 
        'keep *_ALCARECOMuAlCalIsolatedMu_*_*', 
        'keep *_muonCSCDigis_*_*', 
        'keep *_muonDTDigis_*_*', 
        'keep *_muonRPCDigis_*_*', 
        'keep *_dt1DRecHits_*_*', 
        'keep *_dt2DSegments_*_*', 
        'keep *_dt4DSegments_*_*', 
        'keep *_csc2DRecHits_*_*', 
        'keep *_cscSegments_*_*', 
        'keep *_rpcRecHits_*_*'),
    fileName = cms.untracked.string('ALCARECOMuAlCalIsolatedMu.root'),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('StreamALCARECOMuAlCalIsolatedMu'),
        dataTier = cms.untracked.string('ALCARECO')
    )
)

# Path and EndPath definitions
process.endjob_step = cms.Path(process.endOfProcess)
process.ALCARECOStreamMuAlStandAloneCosmicsOutPath = cms.EndPath(process.ALCARECOStreamMuAlStandAloneCosmics)
process.ALCARECOStreamHcalCalHOCosmicsOutPath = cms.EndPath(process.ALCARECOStreamHcalCalHOCosmics)
process.ALCARECOStreamTkAlCosmics0TOutPath = cms.EndPath(process.ALCARECOStreamTkAlCosmics0T)
process.ALCARECOStreamTkAlCosmicsHLTOutPath = cms.EndPath(process.ALCARECOStreamTkAlCosmicsHLT)
process.ALCARECOStreamRpcCalHLTOutPath = cms.EndPath(process.ALCARECOStreamRpcCalHLT)
process.ALCARECOStreamMuAlGlobalCosmicsOutPath = cms.EndPath(process.ALCARECOStreamMuAlGlobalCosmics)
process.ALCARECOStreamMuAlCalIsolatedMuOutPath = cms.EndPath(process.ALCARECOStreamMuAlCalIsolatedMu)

# Schedule definition
process.schedule = cms.Schedule(process.endjob_step,process.ALCARECOStreamMuAlStandAloneCosmicsOutPath,process.ALCARECOStreamHcalCalHOCosmicsOutPath,process.ALCARECOStreamTkAlCosmics0TOutPath,process.ALCARECOStreamTkAlCosmicsHLTOutPath,process.ALCARECOStreamRpcCalHLTOutPath,process.ALCARECOStreamMuAlGlobalCosmicsOutPath,process.ALCARECOStreamMuAlCalIsolatedMuOutPath)
