# Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>
# Part of the MC-truth-graph prototype - under heavy development, not yet open
# to external contributions (see PhysicsTools/TruthInfo/README.md).

# DQM harvesting for the Branch performance-plot validators: turns the booked
# numerator/denominator histograms into reproduction-efficiency plots vs
# eta/pt/energy, in the same fashion as the standard HGCAL/tracking post-processors
# (DQMGenericClient). Profiles (purity/completeness/response) are produced directly
# by the analyzer and need no harvesting.

import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

_branchEfficiency = cms.vstring(
    "efficiency_eta 'Branch reproduction efficiency vs #eta;#eta;efficiency' effnum_eta denom_eta",
    "efficiency_pt 'Branch reproduction efficiency vs p_{T};p_{T} [GeV];efficiency' effnum_pt denom_pt",
    "efficiency_energy 'Branch reproduction efficiency vs E;E [GeV];efficiency' effnum_energy denom_energy",
    # "Other way around": fraction of objects whose best hit-matched Branch is the
    # natural (trackId-seeded) one, from the validator's selfmatch_* numerators.
    "selfmatchrate_eta 'Best Branch is the natural one vs #eta;#eta;self-match rate' selfmatch_eta denom_eta",
    "selfmatchrate_pt 'Best Branch is the natural one vs p_{T};p_{T} [GeV];self-match rate' selfmatch_pt denom_pt",
)

branchHGCalPostProcessor = DQMEDHarvester(
    "DQMGenericClient",
    subDirs=cms.untracked.vstring(
        "HGCAL/BranchValidator/CaloParticle",
        "HGCAL/BranchValidator/SimCluster",
    ),
    efficiency=_branchEfficiency,
    resolution=cms.vstring(),
    verbose=cms.untracked.uint32(0),
    outputFileName=cms.untracked.string(""),
)

# Tracking: "Branch reproduces the TrackingParticle track->truth assignment"
# efficiency vs eta/pt (energy is meaningless for the tracker).
_branchTrackingEfficiency = cms.vstring(
    "efficiency_eta 'Branch reproduction efficiency vs #eta;#eta;efficiency' effnum_eta denom_eta",
    "efficiency_pt 'Branch reproduction efficiency vs p_{T};p_{T} [GeV];efficiency' effnum_pt denom_pt",
)

branchTrackingPostProcessor = DQMEDHarvester(
    "DQMGenericClient",
    subDirs=cms.untracked.vstring(
        "Tracking/BranchValidator/TrackingParticle",
    ),
    efficiency=_branchTrackingEfficiency,
    resolution=cms.vstring(),
    verbose=cms.untracked.uint32(0),
    outputFileName=cms.untracked.string(""),
)

# Generic reco-side validators (BranchRecoValidator): efficiency/duplicate over the
# branches, fake-rate/merge-rate over the reco objects, vs eta and the second axis
# (pt for tracks, energy for tracksters). All are num/den ratios via DQMGenericClient.
def _recoSideEfficiency(xName):
    return cms.vstring(
        "efficiency_eta 'Branch reco efficiency vs #eta;#eta;efficiency' effnum_eta denom_eta",
        "efficiency_%s 'Branch reco efficiency;;efficiency' effnum_%s denom_%s" % (xName, xName, xName),
        "duplicate_eta 'Branch duplicate rate vs #eta;#eta;duplicate' dupnum_eta denom_eta",
        "duplicate_%s 'Branch duplicate rate;;duplicate' dupnum_%s denom_%s" % (xName, xName, xName),
        "fakerate_eta 'Reco fake rate vs #eta;#eta;fake rate' fakenum_eta recodenom_eta",
        "fakerate_%s 'Reco fake rate;;fake rate' fakenum_%s recodenom_%s" % (xName, xName, xName),
        "mergerate_eta 'Reco merge rate vs #eta;#eta;merge rate' mergenum_eta recodenom_eta",
        "mergerate_%s 'Reco merge rate;;merge rate' mergenum_%s recodenom_%s" % (xName, xName, xName),
    )

branchTrackRecoPostProcessor = DQMEDHarvester(
    "DQMGenericClient",
    subDirs=cms.untracked.vstring("Tracking/BranchValidator/recoTrack"),
    efficiency=_recoSideEfficiency("pt"),
    resolution=cms.vstring(),
    verbose=cms.untracked.uint32(0),
    outputFileName=cms.untracked.string(""),
)

branchTracksterRecoPostProcessor = DQMEDHarvester(
    "DQMGenericClient",
    subDirs=cms.untracked.vstring("HGCAL/BranchValidator/Trackster"),
    efficiency=_recoSideEfficiency("energy"),
    resolution=cms.vstring(),
    verbose=cms.untracked.uint32(0),
    outputFileName=cms.untracked.string(""),
)

truthGraphDQMHarvesting = cms.Sequence(branchHGCalPostProcessor + branchTrackingPostProcessor)

# Opt-in harvesting for the experimental reco-side validators (see
# truthGraphRecoSideValidationSequence in truthGraphValidation_cff): pair with that
# sequence only once a disjoint antichain reference is configured.
truthGraphRecoSideHarvesting = cms.Sequence(branchTrackRecoPostProcessor + branchTracksterRecoPostProcessor)
