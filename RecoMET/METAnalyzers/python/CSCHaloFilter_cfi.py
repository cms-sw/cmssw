import FWCore.ParameterSet.Config as cms

from TrackingTools.TrackAssociator.default_cfi import *


CSCBasedHaloFilter = cms.EDFilter("CSCHaloFilter",

                                  BeamHaloSummaryLabel = cms.InputTag("BeamHaloSummary"),
                                  ### Do you want to filter based on BeamHaloSummary::CSCLooseHaloId() ( ~90% eff, 1E-3 mistag rate) 
                                  FilterCSCLoose = cms.bool(False),
                                  ### Do you want to filter based on BeamHaloSummary::CSCTightHaloId() ( ~65% eff, <1E-5 mistag rate)
                                  FilterCSCTight = cms.bool(False),


                                  #############  For Use Only if FilterCSCLoose and FilterCSCTight are false
                                  #
                                  #
                                  #
                                  #

                                  ### Do you want to use L1 CSC BeamHalo Trigger to identify halo? (For < 36X, this requires the RAW-DIGI )
                                  FilterTriggerLevel = cms.bool(True),
                                  ### Do you want to use early ALCT Digis to identify halo? (requires DIGI data tier)
                                  FilterDigiLevel = cms.bool(True),
                                  ### Do you want to use halo-like CSC cosmic reconstructed tracks to identify halo?   
                                  FilterRecoLevel = cms.bool(True),
                                  
                                  # L1
                                  L1MuGMTReadoutLabel = cms.InputTag("gtDigis"),
                                  # Chamber Level Trigger Primitive
                                  ALCTDigiLabel = cms.InputTag("muonCSCDigis","MuonCSCALCTDigi"),
                                  # RecHit Level
                                  CSCRecHitLabel = cms.InputTag("csc2DRecHits"),
                                  # Higher Level Reco
                                  CSCSegmentLabel= cms.InputTag("cscSegments"),
                                  SACosmicMuonLabel= cms.InputTag("cosmicMuons"),
                                  CollisionMuonLabel = cms.InputTag("muons"),
                                  CSCHaloDataLabel = cms.InputTag("CSCHaloData"),
                                  
                                  ###### Cut Parameters
                                  ### minimum delta-eta between innermost-outermost CSC track rechit 
                                  Deta = cms.double(0.1),
                                  ### maximum delta-phi between innermost-outermost CSC track rechit 
                                  Dphi = cms.double(1.00),
                                  ### maximum Chi-Square of CSC cosmic track
                                  NormChi2  = cms.double(8.),
                                  #InnerRMin = cms.double(140.),
                                  #OuterRMin = cms.double(140.),
                                  #InnerRMax = cms.double(310.),
                                  #OuterRMax = cms.double(310.),
                                  ### minimum radius of innermost CSC cosmic track rechit
                                  InnerRMin = cms.double(0.),
                                  ### minimum radius of outermost CSC cosmic track rechit
                                  OuterRMin = cms.double(0.),
                                  ### maximum radius of innermost CSC cosmic track rechit
                                  InnerRMax = cms.double(99999.),
                                  ### maximum radius of outermost CSC cosmic track rechit
                                  OuterRMax = cms.double(99999.),
                                  ### lower edge of theta exclusion window of CSC cosmic track
                                  MinOuterMomentumTheta = cms.double(.10),
                                  ### higher edge of theta exclusion window of CSC cosmic track
                                  MaxOuterMomentumTheta = cms.double(3.0),
                                  ### maximum dr/dz calculated from innermost and outermost rechit of CSC cosmic track
                                  MaxDROverDz = cms.double(0.13),
                                  
                                  ### Phi window for matching collision muon rechits to L1 Halo Triggers
                                  MatchingDPhiThreshold = cms.double(0.18),
                                  ### Eta window for matching collision muon rechits to L1 Halo Triggers
                                  MatchingDEtaThreshold = cms.double(0.4),
                                  ### Wire window for matching collision muon rechits to earl ALCT Digis  
                                  MatchingDWireThreshold= cms.int32(5),

                                  ### Min number of L1 Halo Triggers required to call event "halo" (requires FilterTriggerLevel=True) 
                                  MinNumberOfHaloTriggers = cms.untracked.int32(1),
                                  ### Min number of early ALCT Digis required to call event "halo" (requires FilterDigiLevel =True)
                                  MinNumberOfOutOfTimeDigis = cms.untracked.int32(1),
                                  ### Min number of halo-like CSC cosmic tracks to call event "halo" (requires FilterRecoLevel =True)
                                  MinNumberOfHaloTracks = cms.untracked.int32(1),
                                  
                                  # If this is MC, the expected collision bx for ALCT Digis will be 6 instead of 3
                                  ExpectedBX = cms.int32(3),
                                  TrackAssociatorParameters = TrackAssociatorParameterBlock.TrackAssociatorParameters
                                  )
                             

###CSC Loose Only
CSCLooseHaloFilter = CSCBasedHaloFilter.clone()
CSCLooseHaloFilter.FilterCSCLoose = True
CSCLooseHaloFilter.FilterCSCTight = False

###CSC Tight Only
CSCTightHaloFilter = CSCBasedHaloFilter.clone()
CSCTightHaloFilter.FilterCSCLoose = False
CSCTightHaloFilter.FilterCSCTight = True

###Trigger Level Only###
CSCHaloFilterTriggerLevel = CSCBasedHaloFilter.clone()
CSCHaloFilterTriggerLevel.FilterRecoLevel = False
CSCHaloFilterTriggerLevel.FilterDigiLevel = False

###Reco Level Only ####
CSCHaloFilterRecoLevel = CSCBasedHaloFilter.clone()
CSCHaloFilterRecoLevel.FilterTriggerLevel = False
CSCHaloFilterRecoLevel.FilterDigiLevel = False

### Digi Level Only ###
CSCHaloFilterDigiLevel = CSCBasedHaloFilter.clone()
CSCHaloFilterDigiLevel.FilterTriggerLevel = False
CSCHaloFilterDigiLevel.FilterRecoLevel = False

### Reco AND Trigger Level ###
CSCHaloFilterRecoAndTriggerLevel = CSCBasedHaloFilter.clone()
CSCHaloFilterRecoAndTriggerLevel.FilterDigiLevel = False
### Digi AND Trigger Level ###
CSCHaloFilterDigiAndTriggerLevel = CSCBasedHaloFilter.clone()
CSCHaloFilterDigiAndTriggerLevel.FilterRecoLevel = False
### Digi AND Reco Level ###
CSCHaloFilterDigiAndRecoLevel = CSCBasedHaloFilter.clone()
CSCHaloFilterDigiAndRecoLevel.FilterTriggerLevel = False

### Reco AND Digi AND Trigger Level ###  (Most Restrictive) 
CSCHaloFilterRecoAndDigiAndTriggerLevel = CSCBasedHaloFilter.clone()

### Sequences ####

### Reco OR Trigger Level ###
CSCHaloFilterRecoOrTriggerLevel = cms.Sequence( CSCHaloFilterTriggerLevel * CSCHaloFilterRecoLevel )

### Digi OR Trigger Level ###
CSCHaloFilterDigiOrTriggerLevel = cms.Sequence( CSCHaloFilterDigiLevel * CSCHaloFilterTriggerLevel )

### Digi OR Reco Level ###
CSCHaloFilterDigiOrRecoLevel = cms.Sequence( CSCHaloFilterDigiLevel * CSCHaloFilterRecoLevel )

### Digi OR Reco OR Trigger Level ###  (Loose Selection)
CSCHaloFilterDigiOrRecoOrTriggerLevel = cms.Sequence( CSCHaloFilterDigiLevel * CSCHaloFilterRecoLevel * CSCHaloFilterTriggerLevel )
#CSCHaloFilterLoose = cms.Sequence(CSCHaloFilterDigiOrRecoOrTriggerLevel)

### (Digi AND Reco) OR (Digi AND Trigger) OR (Reco AND Trigger)###  (Tight Selection)
CSCHaloFilter_DigiAndReco_Or_DigiAndTrigger_Or_RecoAndTrigger = cms.Sequence( CSCHaloFilterRecoAndTriggerLevel *
                                                                              CSCHaloFilterDigiAndTriggerLevel *
                                                                              CSCHaloFilterDigiAndRecoLevel )

#CSCHaloFilterTight = cms.Sequence(CSCHaloFilter_DigiAndReco_Or_DigiAndTrigger_Or_RecoAndTrigger)

