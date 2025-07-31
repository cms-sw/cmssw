import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer

scoutingCollectionMonitor = DQMEDAnalyzer('ScoutingCollectionMonitor',
                                          onlyScouting = cms.bool(False),
                                          onlineMetaDataDigis    = cms.InputTag("onlineMetaDataDigis"),
                                          muons                  = cms.InputTag("hltScoutingMuonPackerNoVtx"),
                                          muonsVtx               = cms.InputTag("hltScoutingMuonPackerVtx"),
                                          electrons              = cms.InputTag("hltScoutingEgammaPacker"),
                                          photons                = cms.InputTag("hltScoutingEgammaPacker"),
                                          pfcands                = cms.InputTag("hltScoutingPFPacker"),
                                          pfjets                 = cms.InputTag("hltScoutingPFPacker"),
                                          tracks                 = cms.InputTag("hltScoutingTrackPacker"),
                                          primaryVertices        = cms.InputTag("hltScoutingPrimaryVertexPacker","primaryVtx"),
                                          displacedVertices      = cms.InputTag("hltScoutingMuonPackerVtx","displacedVtx"),
                                          displacedVerticesNoVtx = cms.InputTag("hltScoutingMuonPackerNoVtx","displacedVtx"),
                                          pfMetPt                = cms.InputTag("hltScoutingPFPacker","pfMetPt"),
                                          pfMetPhi               = cms.InputTag("hltScoutingPFPacker","pfMetPhi"),
                                          rho                    = cms.InputTag("hltScoutingPFPacker","rho"),
                                          topfoldername          = cms.string("HLT/ScoutingOffline/Miscellaneous")
                                          )

