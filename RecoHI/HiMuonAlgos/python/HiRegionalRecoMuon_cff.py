import FWCore.ParameterSet.Config as cms

from RecoMuon.Configuration.RecoMuonPPonly_cff import *
from RecoHI.HiMuonAlgos.hiMuonIterativeTk_cff import *

# pretty much everything is as the pp sequence
hiReMuTracks = "hiGeneralAndRegitMuTracks" 

# global muon track
reglobalMuons = globalMuons.clone(
    TrackerCollectionLabel =  hiReMuTracks
)
# tevMuons tracks
retevMuons    = tevMuons.clone(
    MuonCollectionLabel = "reglobalMuons"
)

# trackquality collections
reglbTrackQual = glbTrackQual.clone(
    InputCollection      = "reglobalMuons",
    InputLinksCollection = "reglobalMuons"
)

#recoMuons
remuons = muons1stStep.clone(
    inputCollectionLabels      = [hiReMuTracks, 'reglobalMuons', 'standAloneMuons:UpdatedAtVtx','retevMuons:firstHit','retevMuons:picky','retevMuons:dyt'],
    globalTrackQualityInputTag = 'reglbTrackQual',
    JetExtractorPSet           = dict( JetCollectionLabel   = "iterativeConePu5CaloJets"),
    TrackExtractorPSet         = dict( inputTrackCollection = hiReMuTracks),
    minPt                      = 0.8
)
remuonEcalDetIds = muonEcalDetIds.clone(
    inputCollection = "remuons"
)
#muons.fillGlobalTrackRefits = False

# deposits
remuIsoDepositTk = muIsoDepositTk.clone(
    inputTags = ["remuons:tracker"]
)
remuIsoDepositJets = muIsoDepositJets.clone(
    inputTags = ["remuons:jets"]
)
remuIsoDepositCalByAssociatorTowers = muIsoDepositCalByAssociatorTowers.clone(
    inputTags = ["remuons:ecal", "remuons:hcal", "remuons:ho"]
)
remuonShowerInformation = muonShowerInformation.clone(
    muonCollection = "remuons"
)
# replace the new names

remuonIdProducerTask      = cms.Task(reglbTrackQual,remuons,remuonEcalDetIds,remuonShowerInformation)
remuIsoDeposits_muonsTask = cms.Task(remuIsoDepositTk,remuIsoDepositCalByAssociatorTowers,remuIsoDepositJets)
remuIsolation_muonsTask   = cms.Task(remuIsoDeposits_muonsTask)
remuIsolationTask         = cms.Task(remuIsolation_muonsTask)
#run this if there are no STA muons in events
muontrackingTask                    = cms.Task(standAloneMuonSeedsTask , standAloneMuons , hiRegitMuTrackingTask , reglobalMuons)

#the default setting assumes the STA is already in the event
muontracking_reTask                 = cms.Task(hiRegitMuTrackingTask , reglobalMuons)
muontracking_with_TeVRefinement_reTask  = cms.Task(muontracking_reTask , retevMuons)
muonreco_reTask                     = cms.Task(muontracking_reTask , remuonIdProducerTask)
muonreco_re                         = cms.Sequence(muonreco_reTask)
muonrecowith_TeVRefinemen_reTask    = cms.Task(muontracking_with_TeVRefinement_reTask , remuonIdProducerTask)
muonrecowith_TeVRefinemen_re        = cms.Sequence(muonrecowith_TeVRefinemen_reTask)
muonreco_plus_isolation_reTask      = cms.Task(muonrecowith_TeVRefinemen_reTask , remuIsolationTask)
muonreco_plus_isolation_re          = cms.Sequence(muonreco_plus_isolation_reTask)

reMuonTrackRecoPbPb                 = cms.Sequence(muontracking_reTask)
# HI muon sequence (passed to RecoHI.Configuration.Reconstruction_HI_cff)
regionalMuonRecoPbPb                      = cms.Sequence(muonreco_plus_isolation_reTask)
