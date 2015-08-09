import FWCore.ParameterSet.Config as cms

from RecoMuon.Configuration.RecoMuonPPonly_cff import *

hiTracks = 'hiGeneralTracks' #heavy ion track label

# replace with heavy ion track label
# IMPORTANT
# for now we clone muons1stStep into muons, and the final muon collection is named muonHL.
# for something final we'll need to clone muons1stStep into something like muons, and the final collection will have to be named muons.
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

# turn off calo muons for timing considerations
muons.minPCaloMuon = cms.double( 1.0E9 )

# high level reco
from RecoMuon.MuonIdentification.muons_cfi import muons as muonsHL
muonsHL.InputMuons = cms.InputTag("muons")
muonsHL.PFCandidates = cms.InputTag("particleFlowTmp")
muonsHL.FillDetectorBasedIsolation = cms.bool(False)
muonsHL.FillPFIsolation = cms.bool(False)
muonsHL.FillSelectorMaps = cms.bool(True)
muonsHL.FillShoweringInfo = cms.bool(False)
muonsHL.FillCosmicsIdMap = cms.bool(False)
muidTrackerMuonArbitrated.inputMuonCollection = cms.InputTag("muons")
muidAllArbitrated.inputMuonCollection = cms.InputTag("muons")
muidGlobalMuonPromptTight.inputMuonCollection = cms.InputTag("muons")
muidTMLastStationLoose.inputMuonCollection = cms.InputTag("muons")
muidTMLastStationTight.inputMuonCollection = cms.InputTag("muons")
muidTM2DCompatibilityLoose.inputMuonCollection = cms.InputTag("muons")
muidTM2DCompatibilityTight.inputMuonCollection = cms.InputTag("muons")
muidTMOneStationLoose.inputMuonCollection = cms.InputTag("muons")
muidTMOneStationTight.inputMuonCollection = cms.InputTag("muons")
muidTMLastStationOptimizedLowPtLoose.inputMuonCollection = cms.InputTag("muons")
muidTMLastStationOptimizedLowPtTight.inputMuonCollection = cms.InputTag("muons")
muidGMTkChiCompatibility.inputMuonCollection = cms.InputTag("muons")
muidGMStaChiCompatibility.inputMuonCollection = cms.InputTag("muons")
muidGMTkKinkTight.inputMuonCollection = cms.InputTag("muons")
muidTMLastStationAngLoose.inputMuonCollection = cms.InputTag("muons")
muidTMLastStationAngTight.inputMuonCollection = cms.InputTag("muons")
muidTMOneStationAngLoose.inputMuonCollection = cms.InputTag("muons")
muidTMOneStationAngTight.inputMuonCollection = cms.InputTag("muons")
muidRPCMuLoose.inputMuonCollection = cms.InputTag("muons")
muonRecoHighLevelPbPb = cms.Sequence(muonSelectionTypeSequence+muonsHL)

# HI muon sequence (passed to RecoHI.Configuration.Reconstruction_HI_cff)

muonRecoPbPb = cms.Sequence(muonreco_plus_isolation_PbPb)

