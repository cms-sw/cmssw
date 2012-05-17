import FWCore.ParameterSet.Config as cms

ecalRecHitFlag_kGood=0                     # channel ok, the energy and time measurement are reliable
ecalRecHitFlag_kPoorReco=1                 # the energy is available from the UncalibRecHit, but approximate (bad shape, large chi2)
ecalRecHitFlag_kOutOfTime=2                # the energy is available from the UncalibRecHit (sync reco), but the event is out of time
ecalRecHitFlag_kFaultyHardware=3           # The energy is available from the UncalibRecHit, channel is faulty at some hardware level (e.g. noisy)
ecalRecHitFlag_kNoisy=4                    # the channel is very noisy
ecalRecHitFlag_kPoorCalib=5                # the energy is available from the UncalibRecHit, but the calibration of the channel is poor
ecalRecHitFlag_kSaturated=6                # saturated channel (recovery not tried)
ecalRecHitFlag_kLeadingEdgeRecovered=7     # saturated channel: energy estimated from the leading edge before saturation
ecalRecHitFlag_kNeighboursRecovered=8      # saturated/isolated dead: energy estimated from neighbours
ecalRecHitFlag_kTowerRecovered=9           # channel in TT with no data link, info retrieved from Trigger Primitive
ecalRecHitFlag_kDead=10                    # channel is dead and any recovery fails
ecalRecHitFlag_kKilled=11                  # MC only flag: the channel is killed in the real detector
ecalRecHitFlag_kTPSaturated=12             # the channel is in a region with saturated TP
ecalRecHitFlag_kL1SpikeFlag=13             # the channel is in a region with TP with sFGVB = 0
ecalRecHitFlag_kWeird=14                   # the signal is believed to originate from an anomalous deposit (spike) 
ecalRecHitFlag_kDiWeird=15                 # the signal is anomalous, and neighbors another anomalous signal  
                            
ecalRecHitFlag_kUnknown=16                 # to ease the interface with functions returning flags. 
