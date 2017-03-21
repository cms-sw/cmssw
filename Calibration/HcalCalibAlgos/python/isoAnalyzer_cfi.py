import FWCore.ParameterSet.Config as cms

HcalIsoTrkAnalyzer = cms.EDAnalyzer("HcalIsoTrkAnalyzer",
                                    Triggers          = cms.vstring("HLT_PFJet40","HLT_PFJet60","HLT_PFJet80","HLT_PFJet140","HLT_PFJet200","HLT_PFJet260","HLT_PFJet320","HLT_PFJet400","HLT_PFJet450","HLT_PFJet500"),
                                    ProcessName       = cms.string("HLT"),
                                    L1Filter          = cms.string(""),
                                    L2Filter          = cms.string("L2Filter"),
                                    L3Filter          = cms.string("Filter"),
# following 10 parameters are parameters to select good tracks
                                    TrackQuality      = cms.string("highPurity"),
                                    MinTrackPt        = cms.double(1.0),
                                    MaxDxyPV          = cms.double(0.02),
                                    MaxDzPV           = cms.double(0.02),
                                    MaxChi2           = cms.double(5.0),
                                    MaxDpOverP        = cms.double(0.1),
                                    MinOuterHit       = cms.int32(4),
                                    MinLayerCrossed   = cms.int32(8),
                                    MaxInMiss         = cms.int32(0),
                                    MaxOutMiss        = cms.int32(0),
# Minimum momentum of selected isolated track and signal zone
                                    MinimumTrackP     = cms.double(20.0),
                                    ConeRadius        = cms.double(34.98),
# signal zone in ECAL and MIP energy cutoff
                                    ConeRadiusMIP     = cms.double(14.0),
                                    MaximumEcalEnergy = cms.double(2.0),
# following 4 parameters are for isolation cuts and described in the code
                                    MaxTrackP         = cms.double(8.0),
                                    SlopeTrackP       = cms.double(0.05090504066),
                                    IsolationEnergyStr= cms.double(2.0),
                                    IsolationEnergySft= cms.double(10.0),
# various labels for collections used in the code
                                    TriggerEventLabel = cms.InputTag("hltTriggerSummaryAOD","","HLT"),
                                    TriggerResultLabel= cms.InputTag("TriggerResults","","HLT"),
                                    TrackLabel        = cms.string("generalTracks"),
                                    VertexLabel       = cms.string("offlinePrimaryVertices"),
                                    EBRecHitLabel     = cms.string("EcalRecHitsEB"),
                                    EERecHitLabel     = cms.string("EcalRecHitsEE"),
                                    HBHERecHitLabel   = cms.string("hbhereco"),
                                    BeamSpotLabel     = cms.string("offlineBeamSpot"),
                                    ModuleName        = cms.untracked.string(""),
                                    ProducerName      = cms.untracked.string(""),
#  Various flags used for selecting tracks, choice of energy Method2/0
#  Data type 0/1 for single jet trigger or others
                                    IgnoreTriggers    = cms.untracked.bool(False),
                                    UseRaw            = cms.untracked.bool(False),
                                    HcalScale         = cms.untracked.double(1.0),
                                    DataType          = cms.untracked.int32(0),
                                    OutMode           = cms.untracked.int32(11),
)
