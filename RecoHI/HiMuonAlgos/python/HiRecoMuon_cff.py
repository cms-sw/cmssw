import FWCore.ParameterSet.Config as cms

from RecoMuon.Configuration.RecoMuonPPonly_cff import *

hiTracks = 'hiGeneralTracks' #heavy ion track label

# replace with heavy ion track label
hiMuons1stStep = muons1stStep.clone(
    inputCollectionLabels = [hiTracks, 'globalMuons', 'standAloneMuons:UpdatedAtVtx','tevMuons:firstHit','tevMuons:picky','tevMuons:dyt'],
    inputCollectionTypes  = ['inner tracks', 'links', 'outer tracks','tev firstHit', 'tev picky', 'tev dyt'],
    TrackExtractorPSet    = dict(inputTrackCollection = hiTracks),
    minPt                 = 0.8,
    #iso deposits are not used in HI
    writeIsoDeposits      = False,
    #fillGlobalTrackRefits = False
)

muonEcalDetIds.inputCollection = "hiMuons1stStep"
muIsoDepositTk.inputTags       = ["hiMuons1stStep:tracker"]
muIsoDepositJets.inputTags     = ["hiMuons1stStep:jets"]
muIsoDepositCalByAssociatorTowers.inputTags = ["hiMuons1stStep:ecal", "hiMuons1stStep:hcal", "hiMuons1stStep:ho"]
muonShowerInformation.muonCollection        = "hiMuons1stStep"

#don't modify somebody else's sequence, create a new one if needed
#standalone muon tracking is already done... so remove standalonemuontracking from muontracking
from FWCore.ParameterSet.SequenceTypes import ModuleNodeVisitor
_excludes=[]
_visitor=ModuleNodeVisitor(_excludes)
standalonemuontracking.visit(_visitor)
displacedGlobalMuonTracking.visit(_visitor)
muonreco_plus_isolation_PbPbTask = muonreco_plus_isolationTask.copyAndExclude(_excludes)

muonreco_plus_isolation_PbPbTask.replace(muons1stStep, hiMuons1stStep)
#iso deposits are not used in HI
muonreco_plus_isolation_PbPbTask.remove(muIsoDeposits_muonsTask)
muonreco_plus_isolation_PbPb = cms.Sequence(muonreco_plus_isolation_PbPbTask)

globalMuons.TrackerCollectionLabel = hiTracks

# replace with heavy ion jet label
hiMuons1stStep.JetExtractorPSet.JetCollectionLabel = "iterativeConePu5CaloJets"

# turn off calo muons for timing considerations
hiMuons1stStep.minPCaloMuon = 1.0E9

# high level reco
from RecoMuon.MuonIdentification.muons_cfi import muons
muons = muons.clone(
    InputMuons          = "hiMuons1stStep",
    PFCandidates        = "particleFlowTmp",
    FillDetectorBasedIsolation = False,
    FillPFIsolation     = False,
    FillSelectorMaps    = False,
    FillShoweringInfo   = False,
    FillCosmicsIdMap    = False,
    vertices            = "hiSelectedVertex"
)
muonRecoHighLevelPbPbTask = cms.Task(muons)

# HI muon sequence (passed to RecoHI.Configuration.Reconstruction_HI_cff)
muonRecoPbPbTask = cms.Task(muonreco_plus_isolation_PbPbTask)
