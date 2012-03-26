import FWCore.ParameterSet.Config as cms

from RecoMuon.Configuration.RecoMuonPPonly_cff import *
from RecoHI.HiMuonAlgos.hiMuonIterativeTk_cff import *

# pretty much everything is as the pp sequence
hiReMuTracks = 'hiGeneralAndRegitMuTracks' #'hiRegitMuGeneralTracks'

# global muon track
reglobalMuons = globalMuons.clone()
reglobalMuons.TrackerCollectionLabel                                     = hiReMuTracks

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
muonIdProducerSequence.replace(glbTrackQual,reglbTrackQual)
muonIdProducerSequence.replace(muons1stStep,remuons)
muonIdProducerSequence.replace(tevMuons,retevMuons)
muonIdProducerSequence.replace(calomuons,recalomuons)
muonIdProducerSequence.replace(muonEcalDetIds,remuonEcalDetIds)
muonIdProducerSequence.replace(muonShowerInformation,remuonShowerInformation)
muIsolation.replace(muIsoDepositTk,remuIsoDepositTk)
muIsolation.replace(muIsoDepositJets,remuIsoDepositJets)
muIsolation.replace(muIsoDepositCalByAssociatorTowers,remuIsoDepositCalByAssociatorTowers)

#run this if there are no STA muons in events
muontracking                        = cms.Sequence(standAloneMuonSeeds * standAloneMuons * hiRegitMuTracking * reglobalMuons)

#the default setting assumes the STA is already in the event
muontracking_re                     = cms.Sequence(hiRegitMuTracking * reglobalMuons)
muontracking_with_TeVRefinement_re  = cms.Sequence(muontracking_re * retevMuons)

muonreco_re                         = cms.Sequence(muontracking_re * muonIdProducerSequence)
muonrecowith_TeVRefinemen_re        = cms.Sequence(muontracking_with_TeVRefinement_re * muonIdProducerSequence)
muonreco_plus_isolation_re          = cms.Sequence(muonrecowith_TeVRefinemen_re * muIsolation)

reMuonTrackRecoPbPb                 = cms.Sequence(muontracking_re)
# HI muon sequence (passed to RecoHI.Configuration.Reconstruction_HI_cff)
reMuonRecoPbPb                      = cms.Sequence(muonreco_plus_isolation_re)
