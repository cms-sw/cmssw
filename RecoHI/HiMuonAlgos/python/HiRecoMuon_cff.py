import FWCore.ParameterSet.Config as cms

from RecoMuon.Configuration.RecoMuonPPonly_cff import *

hiTracks = 'hiGeneralTracks' #heavy ion track label

# replace with heavy ion track label
muons = muons1stStep.clone()
muons.inputCollectionLabels = [hiTracks, 'globalMuons', 'standAloneMuons:UpdatedAtVtx','tevMuons:firstHit','tevMuons:picky','tevMuons:dyt']
muons.inputCollectionTypes = ['inner tracks', 'links', 'outer tracks','tev firstHit', 'tev picky', 'tev dyt']
muons.TrackExtractorPSet.inputTrackCollection = hiTracks
muons.minPt = cms.double(0.8)
#muons.fillGlobalTrackRefits = False
muonEcalDetIds.inputCollection = "muons"

calomuons.inputTracks = hiTracks
calomuons.inputCollection = 'muons'
calomuons.inputMuons = 'muons'
muIsoDepositTk.inputTags = cms.VInputTag(cms.InputTag("muons:tracker"))
muIsoDepositJets. inputTags = cms.VInputTag(cms.InputTag("muons:jets"))
muIsoDepositCalByAssociatorTowers.inputTags = cms.VInputTag(cms.InputTag("muons:ecal"), cms.InputTag("muons:hcal"), cms.InputTag("muons:ho"))

muonShowerInformation.muonCollection = "muons"

#don't modify somebody else's sequence, create a new one if needed
#standalone muon tracking is already done... so remove standalonemuontracking from muontracking
muonreco_plus_isolation_PbPb = muonreco_plus_isolation.copyAndExclude(standalonemuontracking._seq._collection + displacedGlobalMuonTracking._seq._collection)
muonreco_plus_isolation_PbPb.replace(muons1stStep, muons)


globalMuons.TrackerCollectionLabel = hiTracks

# replace with heavy ion jet label
muons.JetExtractorPSet.JetCollectionLabel = cms.InputTag("iterativeConePu5CaloJets")

# turn off calo muons for timing considerations?
#muons.fillCaloCompatibility = cms.bool(False)

# HI muon sequence (passed to RecoHI.Configuration.Reconstruction_HI_cff)

muonRecoPbPb = cms.Sequence(muonreco_plus_isolation_PbPb)

