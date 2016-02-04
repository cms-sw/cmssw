import FWCore.ParameterSet.Config as cms

ecalRecHitFlag_kGood = 0                  # channel ok, the energy and time measurement are reliable
ecalRecHitFlag_kPoorReco = 1              # the energy is available from the UncalibRecHit, but approximate (bad shape, large chi2)
ecalRecHitFlag_kOutOfTime = 2             # the energy is available from the UncalibRecHit (sync reco), but the event is out of time
ecalRecHitFlag_kFaultyHardware = 3        # The energy is available from the UncalibRecHit, channel is faulty at some hardware level (e.g. noisy)
ecalRecHitFlag_kPoorCalib = 4             # the energy is available from the UncalibRecHit, but the calibration of the channel is poor
ecalRecHitFlag_kSaturated = 5             # saturated channel (recovery not tried)
ecalRecHitFlag_kLeadingEdgeRecovered = 6  # saturated channel: energy estimated from the leading edge before saturation
ecalRecHitFlag_kNeighboursRecovered = 7   # saturated/isolated dead: energy estimated from neighbours
ecalRecHitFlag_kTowerRecovered = 8        # channel in TT with no data link, info retrieved from Trigger Primitive
ecalRecHitFlag_kDead = 9
