# Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>

"""RECO-step customise that keeps the truth built at DIGI (the logical graph, the hit
index and the raw graph TruthGraph_mix) in the RECO-tier output, at the requested
verbosity level."""

from PhysicsTools.TruthInfo.truthEventContent_cff import setTruthEventContent


def customise(process, level='compact', includeTrackingHits=True):
    """Persist the DIGI-built truth into the RECO-tier output."""
    return setTruthEventContent(process, level=level, includeTrackingHits=includeTrackingHits)
