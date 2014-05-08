import FWCore.ParameterSet.Config as cms

# -*-TCL-*-
from RecoMuon.MuonIsolationProducers.trackAssociatorBlocks_cff import *
MIsoJetExtractorBlock = cms.PSet(
    MIsoTrackAssociatorJets,
    PrintTimeReport = cms.untracked.bool(False),
    #subtract sumEt of towers falling into the muon-veto region
    ExcludeMuonVeto = cms.bool(True),
    ComponentName = cms.string('JetExtractor'),
    DR_Max = cms.double(1.0),
    PropagatorName = cms.string('SteppingHelixPropagatorAny'),
    ServiceParameters = cms.PSet(
        Propagators = cms.untracked.vstring( 'SteppingHelixPropagatorAny' ),
        RPCLayers = cms.bool( False ),
        UseMuonNavigation = cms.untracked.bool( False )
    ),
    DR_Veto = cms.double(0.1),
    #count a jet if et> threshold
    #note: et from crossed towers and jet towers inside the veto cone are not counted
    Threshold = cms.double(5.0),
    #    InputTag JetCollectionLabel = midPointCone5CaloJets
    #    JetCollectionLabel = cms.InputTag("sisCone5CaloJets")
    JetCollectionLabel = cms.InputTag("ak4CaloJets")
)


