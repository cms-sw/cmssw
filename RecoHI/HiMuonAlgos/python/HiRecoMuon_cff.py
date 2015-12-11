import FWCore.ParameterSet.Config as cms

from RecoMuon.Configuration.RecoMuonPPonly_cff import *

hiTracks = 'hiGeneralTracks' #heavy ion track label

# replace with heavy ion track label
hiMuons1stStep = muons1stStep.clone()
hiMuons1stStep.inputCollectionLabels = [hiTracks, 'globalMuons', 'standAloneMuons:UpdatedAtVtx','tevMuons:firstHit','tevMuons:picky','tevMuons:dyt']
hiMuons1stStep.inputCollectionTypes = ['inner tracks', 'links', 'outer tracks','tev firstHit', 'tev picky', 'tev dyt']
hiMuons1stStep.TrackExtractorPSet.inputTrackCollection = hiTracks
hiMuons1stStep.minPt = cms.double(0.8)
#iso deposits are not used in HI
hiMuons1stStep.writeIsoDeposits = False
#hiMuons1stStep.fillGlobalTrackRefits = False

muonEcalDetIds.inputCollection = "hiMuons1stStep"

calomuons.inputTracks = hiTracks
calomuons.inputCollection = 'hiMuons1stStep'
calomuons.inputMuons = 'hiMuons1stStep'
muIsoDepositTk.inputTags = cms.VInputTag(cms.InputTag("hiMuons1stStep:tracker"))
muIsoDepositJets. inputTags = cms.VInputTag(cms.InputTag("hiMuons1stStep:jets"))
muIsoDepositCalByAssociatorTowers.inputTags = cms.VInputTag(cms.InputTag("hiMuons1stStep:ecal"), cms.InputTag("hiMuons1stStep:hcal"), cms.InputTag("hiMuons1stStep:ho"))

muonShowerInformation.muonCollection = "hiMuons1stStep"

#don't modify somebody else's sequence, create a new one if needed
#standalone muon tracking is already done... so remove standalonemuontracking from muontracking
muonreco_plus_isolation_PbPb = muonreco_plus_isolation.copyAndExclude(standalonemuontracking._seq._collection + displacedGlobalMuonTracking._seq._collection)
muonreco_plus_isolation_PbPb.replace(muons1stStep, hiMuons1stStep)
#iso deposits are not used in HI
muonreco_plus_isolation_PbPb.remove(muIsoDeposits_muons)

globalMuons.TrackerCollectionLabel = hiTracks

# replace with heavy ion jet label
hiMuons1stStep.JetExtractorPSet.JetCollectionLabel = cms.InputTag("iterativeConePu5CaloJets")

# turn off calo muons for timing considerations
hiMuons1stStep.minPCaloMuon = cms.double( 1.0E9 )

# high level reco
from RecoMuon.MuonIdentification.muons_cfi import muons
muons.InputMuons = cms.InputTag("hiMuons1stStep")
muons.PFCandidates = cms.InputTag("particleFlowTmp")
muons.FillDetectorBasedIsolation = cms.bool(False)
muons.FillPFIsolation = cms.bool(False)
muons.FillSelectorMaps = cms.bool(False)
muons.FillShoweringInfo = cms.bool(False)
muons.FillCosmicsIdMap = cms.bool(False)
muonRecoHighLevelPbPb = cms.Sequence(muons)

# HI muon sequence (passed to RecoHI.Configuration.Reconstruction_HI_cff)

muonRecoPbPb = cms.Sequence(muonreco_plus_isolation_PbPb)

