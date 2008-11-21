# Auto generated configuration file
# using: 
# Revision: 1.77.2.1 
# Source: /cvs/CMSSW/CMSSW/Configuration/PyReleaseValidation/python/ConfigBuilder.py,v 
# with command line options: step3 -s ALCA:TkAlCosmics0T+TkAlCosmicsHLT+MuAlStandAloneCosmics+MuAlGlobalCosmics+MuAlBeamHaloOverlaps+HcalCalHOCosmics+DQM --conditions FrontierConditions_GlobalTag,CRAFT_V2P::All --no_exec
import FWCore.ParameterSet.Config as cms

process = cms.Process('ALCA')

# import of standard configurations
##process.load('Configuration/StandardSequences/Services_cff')
process.DQMStore = cms.Service("DQMStore")
process.load('FWCore/MessageService/MessageLogger_cfi')
process.load('Configuration/StandardSequences/MixingNoPileUp_cff')
process.load('Configuration/StandardSequences/GeometryPilot2_cff')
process.load('Configuration/StandardSequences/MagneticField_38T_cff')
process.load('Configuration/StandardSequences/AlCaRecoStreams_cff')
process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
process.load('Configuration/EventContent/EventContent_cff')

process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.2 $'),
    annotation = cms.untracked.string('step3 nevts:1'),
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
    fileNames = cms.untracked.vstring('rfio:/castor/cern.ch/user/e/efe/VALIDATION/CRAFT/promptrecoCosmics_10.root')
)

# Additional output definition
process.ALCARECOStreamMuAlStandAloneCosmics = cms.OutputModule("PoolOutputModule",
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOMuAlStandAloneCosmics')
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
process.ALCARECOStreamMuAlGlobalCosmics = cms.OutputModule("PoolOutputModule",
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOMuAlGlobalCosmics')
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
process.ALCARECOStreamMuAlBeamHaloOverlaps = cms.OutputModule("PoolOutputModule",
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOMuAlBeamHaloOverlaps')
    ),
    outputCommands = cms.untracked.vstring('drop *', 
        'keep *_ALCARECOMuAlBeamHaloOverlaps_*_*', 
        'keep *_muonCSCDigis_*_*', 
        'keep *_csc2DRecHits_*_*', 
        'keep *_cscSegments_*_*'),
    fileName = cms.untracked.string('ALCARECOMuAlBeamHaloOverlaps.root'),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('StreamALCARECOMuAlBeamHaloOverlaps'),
        dataTier = cms.untracked.string('ALCARECO')
    )
)
process.ALCARECOStreamTkAlCosmics0T = cms.OutputModule("PoolOutputModule",
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOTkAlCosmicsCTF0T', 
            'pathALCARECOTkAlCosmicsCosmicTF0T', 
            'pathALCARECOTkAlCosmicsRS0T')
    ),
    outputCommands = cms.untracked.vstring('drop *', 
        'keep *_ALCARECOTkAlCosmics*0T_*_*', 
        'keep L1GlobalTriggerReadoutRecord_gtDigis_*_*', 
        'keep Si*Cluster*_*_*_*', 
        'keep *_MEtoEDMConverter_*_*'),
    fileName = cms.untracked.string('ALCARECOTkAlCosmics0T.root'),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('StreamALCARECOTkAlCosmics0T'),
        dataTier = cms.untracked.string('ALCARECO')
    )
)
process.ALCARECOStreamTkAlCosmicsHLT = cms.OutputModule("PoolOutputModule",
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOTkAlCosmicsCTFHLT', 
            'pathALCARECOTkAlCosmicsCosmicTFHLT', 
            'pathALCARECOTkAlCosmicsRSHLT')
    ),
    outputCommands = cms.untracked.vstring('drop *', 
        'keep *_ALCARECOTkAlCosmics*_*_*', 
        'keep L1GlobalTriggerReadoutRecord_gtDigis_*_*', 
        'keep Si*Cluster*_*_*_*', 
        'keep *_MEtoEDMConverter_*_*'),
    fileName = cms.untracked.string('ALCARECOTkAlCosmicsHLT.root'),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('StreamALCARECOTkAlCosmicsHLT'),
        dataTier = cms.untracked.string('ALCARECO')
    )
)
process.ALCARECOStreamHcalCalHOCosmics = cms.OutputModule("PoolOutputModule",
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOHcalCalHOCosmics')
    ),
    outputCommands = cms.untracked.vstring('drop *', 
        'keep HOCalibVariabless_*_*_*'),
    fileName = cms.untracked.string('ALCARECOHcalCalHOCosmics.root'),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('StreamALCARECOHcalCalHOCosmics'),
        dataTier = cms.untracked.string('ALCARECO')
    )
)

# Other statements
process.GlobalTag.globaltag = 'CRAFT_ALL_V3::All'

# Path and EndPath definitions
process.pathALCARECOTkAlCosmicsCTFHLT = cms.Path(process.seqALCARECOTkAlCosmicsCTFHLT)
process.pathALCARECORpcCalHLT = cms.Path(process.seqALCARECORpcCalHLT)
process.pathALCARECOHcalCalGammaJet = cms.Path(process.seqALCARECOHcalCalGammaJet)
process.pathALCARECOHcalCalHOCosmics = cms.Path(process.seqALCARECOHcalCalHOCosmics)
process.pathALCARECOMuAlBeamHaloOverlaps = cms.Path(process.seqALCARECOMuAlBeamHaloOverlaps)
process.pathALCARECOTkAlCosmicsCosmicTF0THLT = cms.Path(process.seqALCARECOTkAlCosmicsCosmicTF0THLT*process.ALCARECOTkAlCosmicsCosmicTF0TDQM)
process.pathALCARECOMuAlZeroFieldGlobalCosmics = cms.Path(process.seqALCARECOMuAlZeroFieldGlobalCosmics)
process.pathALCARECOTkAlCosmicsCosmicTFHLT = cms.Path(process.seqALCARECOTkAlCosmicsCosmicTFHLT)
process.pathALCARECOSiStripCalMinBias = cms.Path(process.seqALCARECOSiStripCalMinBias)
process.pathALCARECOTkAlCosmicsRS0T = cms.Path(process.seqALCARECOTkAlCosmicsRS0T*process.ALCARECOTkAlCosmicsRS0TDQM)
process.pathALCARECOTkAlMinBias = cms.Path(process.seqALCARECOTkAlMinBias*process.ALCARECOTkAlMinBiasDQM)
process.pathALCARECOTkAlMuonIsolated = cms.Path(process.seqALCARECOTkAlMuonIsolated*process.ALCARECOTkAlMuonIsolatedDQM)
process.pathALCARECOMuAlStandAloneCosmics = cms.Path(process.seqALCARECOMuAlStandAloneCosmics)
process.pathALCARECODQM = cms.Path(process.MEtoEDMConverter)
process.pathALCARECOTkAlZMuMu = cms.Path(process.seqALCARECOTkAlZMuMu*process.ALCARECOTkAlZMuMuDQM)
process.pathALCARECOTkAlUpsilonMuMu = cms.Path(process.seqALCARECOTkAlUpsilonMuMu*process.ALCARECOTkAlUpsilonMuMuDQM)
process.pathALCARECOHcalCalDijets = cms.Path(process.seqALCARECOHcalCalDijets)
process.pathALCARECOTkAlCosmicsCTF0T = cms.Path(process.seqALCARECOTkAlCosmicsCTF0T*process.ALCARECOTkAlCosmicsCTF0TDQM)
process.pathALCARECOMuAlOverlaps = cms.Path(process.seqALCARECOMuAlOverlaps)
process.pathALCARECOTkAlCosmicsRS0THLT = cms.Path(process.seqALCARECOTkAlCosmicsRS0THLT*process.ALCARECOTkAlCosmicsRS0TDQM)
process.pathALCARECOTkAlCosmicsCosmicTF = cms.Path(process.seqALCARECOTkAlCosmicsCosmicTF*process.ALCARECOTkAlCosmicsCosmicTFDQM)
process.pathALCARECOMuAlGlobalCosmics = cms.Path(process.seqALCARECOMuAlGlobalCosmics)
process.pathALCARECOTkAlBeamHalo = cms.Path(process.seqALCARECOTkAlBeamHalo)
process.pathALCARECOTkAlLAS = cms.Path(process.seqALCARECOTkAlLAS)
process.pathALCARECOTkAlCosmicsRS = cms.Path(process.seqALCARECOTkAlCosmicsRS*process.ALCARECOTkAlCosmicsRSDQM)
process.pathALCARECOSiPixelLorentzAngle = cms.Path(process.seqALCARECOSiPixelLorentzAngle)
process.pathALCARECOMuAlBeamHalo = cms.Path(process.seqALCARECOMuAlBeamHalo)
process.pathALCARECOTkAlCosmicsCTF = cms.Path(process.seqALCARECOTkAlCosmicsCTF*process.ALCARECOTkAlCosmicsCTFDQM)
process.pathALCARECOTkAlCosmicsCosmicTF0T = cms.Path(process.seqALCARECOTkAlCosmicsCosmicTF0T*process.ALCARECOTkAlCosmicsCosmicTF0TDQM)
process.pathALCARECOTkAlJpsiMuMu = cms.Path(process.seqALCARECOTkAlJpsiMuMu*process.ALCARECOTkAlJpsiMuMuDQM)
process.pathALCARECOEcalCalElectron = cms.Path(process.seqALCARECOEcalCalElectron)
process.pathALCARECOTkAlCosmicsCTF0THLT = cms.Path(process.seqALCARECOTkAlCosmicsCTF0THLT*process.ALCARECOTkAlCosmicsCTF0TDQM)
process.pathALCARECOMuAlCalIsolatedMu = cms.Path(process.seqALCARECOMuAlCalIsolatedMu)
process.pathALCARECOHcalCalHO = cms.Path(process.seqALCARECOHcalCalHO)
process.pathALCARECOTkAlCosmicsRSHLT = cms.Path(process.seqALCARECOTkAlCosmicsRSHLT)
process.ALCARECOStreamMuAlStandAloneCosmicsOutPath = cms.EndPath(process.ALCARECOStreamMuAlStandAloneCosmics)
process.ALCARECOStreamMuAlGlobalCosmicsOutPath = cms.EndPath(process.ALCARECOStreamMuAlGlobalCosmics)
process.ALCARECOStreamMuAlBeamHaloOverlapsOutPath = cms.EndPath(process.ALCARECOStreamMuAlBeamHaloOverlaps)
process.ALCARECOStreamTkAlCosmics0TOutPath = cms.EndPath(process.ALCARECOStreamTkAlCosmics0T)
process.ALCARECOStreamTkAlCosmicsHLTOutPath = cms.EndPath(process.ALCARECOStreamTkAlCosmicsHLT)
process.ALCARECOStreamHcalCalHOCosmicsOutPath = cms.EndPath(process.ALCARECOStreamHcalCalHOCosmics)

# Use only HLT paths in the current HLT menu
process.ALCARECOTkAlCosmicsCosmicTFHLT.HLTPaths = ['']
process.ALCARECOTkAlCosmicsRSHLT.HLTPaths = ['']
process.isoMuonHLT.HLTPaths = ['HLT_L1MuOpen']

# Schedule definition
process.schedule = cms.Schedule(process.pathALCARECOMuAlStandAloneCosmics,process.pathALCARECOMuAlBeamHaloOverlaps,process.pathALCARECOTkAlCosmicsCTFHLT,process.pathALCARECOTkAlCosmicsCosmicTFHLT,process.pathALCARECOTkAlCosmicsRSHLT,process.pathALCARECOTkAlCosmicsCTF0T,process.pathALCARECOTkAlCosmicsCosmicTF0T,process.pathALCARECOTkAlCosmicsRS0T,process.pathALCARECOHcalCalHOCosmics,process.pathALCARECODQM,process.pathALCARECOMuAlGlobalCosmics,process.ALCARECOStreamMuAlStandAloneCosmicsOutPath,process.ALCARECOStreamMuAlGlobalCosmicsOutPath,process.ALCARECOStreamMuAlBeamHaloOverlapsOutPath,process.ALCARECOStreamTkAlCosmics0TOutPath,process.ALCARECOStreamTkAlCosmicsHLTOutPath,process.ALCARECOStreamHcalCalHOCosmicsOutPath)
