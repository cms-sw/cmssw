# $Id$
import FWCore.ParameterSet.Config as cms

l1tsClient = cms.EDFilter("L1ScalersClient",
  algoMonitorBits = cms.untracked.vint32(54,55,56),
  techMonitorBits = cms.untracked.vint32(1,2,9),
  dqmFolder = cms.untracked.string("L1T/L1Scalers_EvF")
)

