import FWCore.ParameterSet.Config as cms

from RecoMET.METProducers.TCMET_cfi import *

tcMetWithPFclusters = tcMet.clone()

tcMetWithPFclusters.usePFClusters = cms.bool(True)

tcMetSequence = cms.Sequence(tcMet + tcMetWithPFclusters)
