
import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
dqmElectronGeneralAnalysis = DQMEDAnalyzer('ElectronGeneralAnalyzer',

    Verbosity = cms.untracked.int32(0),
    FinalStep = cms.string("AtJobEnd"),
    InputFile = cms.string(""),
    OutputFile = cms.string(""),
    InputFolderName = cms.string("Egamma/Electrons/General"),
    OutputFolderName = cms.string("Egamma/Electrons/General"),
    
    ElectronCollection = cms.InputTag("gedGsfElectrons"),
    MatchingObjectCollection = cms.InputTag("mergedSuperClusters"),
    TrackCollection = cms.InputTag("generalTracks"),
    GsfTrackCollection = cms.InputTag("electronGsfTracks"),
    VertexCollection = cms.InputTag("offlinePrimaryVertices"),
    BeamSpot = cms.InputTag("offlineBeamSpot"),
)

from Configuration.Eras.Modifier_phase2_hgcal_cff import phase2_hgcal
phase2_hgcal.toModify( dqmElectronGeneralAnalysis, ElectronCollection = cms.InputTag("ecalDrivenGsfElectrons") )
