import FWCore.ParameterSet.Config as cms

from RecoMuon.Configuration.RecoMuonPPonly_cff import *
from RecoHI.HiMuonAlgos.hiMuonIterativeTk_cff import *

# pretty much everything is as the pp sequence
hiReMuTracks = "hiGeneralAndRegitMuTracks" 

# global muon track
reglobalMuons = globalMuons.clone()
reglobalMuons.TrackerCollectionLabel =  hiReMuTracks

# tevMuons tracks
retevMuons    = tevMuons.clone()
retevMuons.MuonCollectionLabel = cms.InputTag("reglobalMuons")


# trackquality collections
reglbTrackQual = glbTrackQual.clone()
reglbTrackQual.InputCollection      = cms.InputTag("reglobalMuons")
reglbTrackQual.InputLinksCollection = cms.InputTag("reglobalMuons")


#recoMuons
remuons       = muons1stStep.clone()
remuons.inputCollectionLabels                   = [hiReMuTracks, 'reglobalMuons', 'standAloneMuons:UpdatedAtVtx','retevMuons:firstHit','retevMuons:picky','retevMuons:dyt']
remuons.globalTrackQualityInputTag              = cms.InputTag('reglbTrackQual')
remuons.JetExtractorPSet.JetCollectionLabel     = cms.InputTag("iterativeConePu5CaloJets")
remuons.TrackExtractorPSet.inputTrackCollection = hiReMuTracks
remuons.minPt = cms.double(0.8)

remuonEcalDetIds = muonEcalDetIds.clone()
remuonEcalDetIds.inputCollection                = "remuons"

#muons.fillGlobalTrackRefits = False
# calomuons
recalomuons   = calomuons.clone()
recalomuons.inputTracks          = hiReMuTracks
recalomuons.inputCollection      = 'remuons'
recalomuons.inputMuons           = 'remuons'

# deposits
remuIsoDepositTk = muIsoDepositTk.clone()
remuIsoDepositTk.inputTags                    = cms.VInputTag(cms.InputTag("remuons:tracker"))
remuIsoDepositJets = muIsoDepositJets.clone()
remuIsoDepositJets.inputTags                  = cms.VInputTag(cms.InputTag("remuons:jets"))
remuIsoDepositCalByAssociatorTowers = muIsoDepositCalByAssociatorTowers.clone()
remuIsoDepositCalByAssociatorTowers.inputTags = cms.VInputTag(cms.InputTag("remuons:ecal"), cms.InputTag("remuons:hcal"), cms.InputTag("remuons:ho"))

remuonShowerInformation                       = muonShowerInformation.clone()
remuonShowerInformation.muonCollection        = "remuons"

# replace the new names

remuonIdProducerSequence = cms.Sequence(reglbTrackQual*remuons*recalomuons*remuonEcalDetIds*remuonShowerInformation)
remuIsoDeposits_muons    = cms.Sequence(remuIsoDepositTk+remuIsoDepositCalByAssociatorTowers+remuIsoDepositJets)
remuIsolation_muons      = cms.Sequence(remuIsoDeposits_muons)
remuIsolation            = cms.Sequence(remuIsolation_muons)
#run this if there are no STA muons in events
muontracking                        = cms.Sequence(standAloneMuonSeeds * standAloneMuons * hiRegitMuTracking * reglobalMuons)

#the default setting assumes the STA is already in the event
muontracking_re                     = cms.Sequence(hiRegitMuTracking * reglobalMuons)
muontracking_with_TeVRefinement_re  = cms.Sequence(muontracking_re * retevMuons)

muonreco_re                         = cms.Sequence(muontracking_re * remuonIdProducerSequence)
muonrecowith_TeVRefinemen_re        = cms.Sequence(muontracking_with_TeVRefinement_re * remuonIdProducerSequence)
muonreco_plus_isolation_re          = cms.Sequence(muonrecowith_TeVRefinemen_re * remuIsolation)

reMuonTrackRecoPbPb                 = cms.Sequence(muontracking_re)
# HI muon sequence (passed to RecoHI.Configuration.Reconstruction_HI_cff)
regionalMuonRecoPbPb                      = cms.Sequence(muonreco_plus_isolation_re)
