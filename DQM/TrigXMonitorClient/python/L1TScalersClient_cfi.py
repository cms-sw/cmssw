# $Id: L1TScalersClient_cfi.py,v 1.2 2008/09/07 03:17:10 wittich Exp $
import FWCore.ParameterSet.Config as cms

l1tsClient = cms.EDAnalyzer("L1ScalersClient",
  algoMonitorBits = cms.untracked.vint32(54,55,56),
  techMonitorBits = cms.untracked.vint32(1,2,9),
  dqmFolder = cms.untracked.string("L1T/L1Scalers_EvF")
)

