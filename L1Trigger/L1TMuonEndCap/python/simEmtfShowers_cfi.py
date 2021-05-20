import FWCore.ParameterSet.Config as cms

from L1Trigger.L1TMuonEndCap.simEmtfShowersDef_cfi import simEmtfShowersDef

## producer for simulation
simEmtfShowers = simEmtfShowersDef.clone()

## producer for re-emulation on unpacked CSC shower data
simEmtfShowersData = simEmtfShowers.clone()
