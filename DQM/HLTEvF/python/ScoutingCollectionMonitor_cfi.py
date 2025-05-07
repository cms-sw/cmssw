import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer



scoutingCollectionMonitor = DQMEDAnalyzer('ScoutingCollectionMonitor',
                                      triggerresults   = cms.InputTag("TriggerResults", "", "HLT"),
                                      muons             = cms.InputTag("hltScoutingMuonPackerNoVtx"),
                                      electrons         = cms.InputTag("hltScoutingEgammaPacker"),
                                      photons           = cms.InputTag("hltScoutingEgammaPacker"),
                                      pfcands           = cms.InputTag("hltScoutingPFPacker"),
                                      pfjets            = cms.InputTag("hltScoutingPFPacker"),
                                      tracks            = cms.InputTag("hltScoutingTrackPacker"),
                                      primaryVertices   = cms.InputTag("hltScoutingPrimaryVertexPacker","primaryVtx"),
                                      displacedVertices = cms.InputTag("hltScoutingMuonPackerNoVtx","displacedVtx"),
                                      pfMetPt           = cms.InputTag("hltScoutingPFPacker","pfMetPt"),
                                      pfMetPhi          = cms.InputTag("hltScoutingPFPacker","pfMetPhi"),
                                      rho               = cms.InputTag("hltScoutingPFPacker","rho"),
                                  )

