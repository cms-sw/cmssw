import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer

scoutingCollectionMonitor = DQMEDAnalyzer('ScoutingCollectionMonitor',
                                          topfoldername          = cms.string("HLT/ScoutingOffline/Miscellaneous"),
                                          onlyScouting           = cms.bool(False),
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
                                          vmBestTrackIndex       = cms.InputTag('run3ScoutingElectronBestTrack', 'Run3ScoutingElectronBestTrackIndex'),
                                          vmTrkd0                = cms.InputTag('run3ScoutingElectronBestTrack', 'Run3ScoutingElectronTrackd0'),
                                          vmTrkdz                = cms.InputTag('run3ScoutingElectronBestTrack', 'Run3ScoutingElectronTrackdz'),
                                          vmTrkpt                = cms.InputTag('run3ScoutingElectronBestTrack', 'Run3ScoutingElectronTrackpt'),
                                          vmTrketa               = cms.InputTag('run3ScoutingElectronBestTrack', 'Run3ScoutingElectronTracketa'),
                                          vmTrkphi               = cms.InputTag('run3ScoutingElectronBestTrack', 'Run3ScoutingElectronTrackphi'),
                                          vmTrkpMode             = cms.InputTag('run3ScoutingElectronBestTrack', 'Run3ScoutingElectronTrackpMode'),
                                          vmTrketaMode           = cms.InputTag('run3ScoutingElectronBestTrack', 'Run3ScoutingElectronTracketaMode'),
                                          vmTrkphiMode           = cms.InputTag('run3ScoutingElectronBestTrack', 'Run3ScoutingElectronTrackphiMode'),
                                          vmTrkqoverpModeError   = cms.InputTag('run3ScoutingElectronBestTrack', 'Run3ScoutingElectronTrackqoverpModeError'),
                                          vmTrkchi2overndf       = cms.InputTag('run3ScoutingElectronBestTrack', 'Run3ScoutingElectronTrackchi2overndf'),
                                          vmTrkcharge            = cms.InputTag('run3ScoutingElectronBestTrack', 'Run3ScoutingElectronTrackcharge'),
                                          pfRecHitsEB            = cms.InputTag(""),
                                          pfRecHitsEE            = cms.InputTag(""),
                                          pfCleanedRecHitsEB     = cms.InputTag(""),
                                          pfCleanedRecHitsEE     = cms.InputTag(""),
                                          pfRecHitsHBHE          = cms.InputTag(""))

## Add the scouting rechits monitoring (only for 2025, integrated in menu GRun 2025 V1.3)
## See https://its.cern.ch/jira/browse/CMSHLT-3607
from Configuration.Eras.Modifier_run3_scouting_2025_cff import run3_scouting_2025
run3_scouting_2025.toModify(scoutingCollectionMonitor,
                            pfRecHitsEB        = ("hltScoutingRecHitPacker", "EB"),
                            pfRecHitsEE        = ("hltScoutingRecHitPacker", "EE"),
                            pfCleanedRecHitsEB = ("hltScoutingRecHitPacker", "EBCleaned"),
                            pfCleanedRecHitsEE = ("hltScoutingRecHitPacker", "EECleaned"),
                            pfRecHitsHBHE      = ("hltScoutingRecHitPacker", "HBHE"))
