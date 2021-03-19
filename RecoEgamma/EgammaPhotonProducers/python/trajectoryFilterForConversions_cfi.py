import FWCore.ParameterSet.Config as cms

import TrackingTools.TrajectoryFiltering.TrajectoryFilter_cff
TrajectoryFilterForConversions = TrackingTools.TrajectoryFiltering.TrajectoryFilter_cff.CkfBaseTrajectoryFilter_block.clone(
    chargeSignificance  = -1.0,
    minPt               = 0.9,
    minHitsMinPt        = -1,
    ComponentType       = 'CkfBaseTrajectoryFilter',
    maxLostHits         = 1,
    maxNumberOfHits     = -1,
    maxConsecLostHits   = 1,
    nSigmaMinPt         = 5.0,
    minimumNumberOfHits = 3,
    maxCCCLostHits      = 9999,
    minGoodStripCharge  = dict(refToPSet_ = 'SiStripClusterChargeCutNone')
)
