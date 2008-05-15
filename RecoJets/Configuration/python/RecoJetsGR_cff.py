import FWCore.ParameterSet.Config as cms

# Standard set:
from RecoJets.JetProducers.kt4CaloJets_cff import *
from RecoJets.JetProducers.kt6CaloJets_cff import *
from RecoJets.JetProducers.iterativeCone5CaloJets_cff import *
from RecoJets.JetProducers.sisCone5CaloJets_cff import *
from RecoJets.JetProducers.sisCone7CaloJets_cff import *
# $Id: RecoJetsGR.cff,v 1.3 2008/04/30 14:23:31 arizzi Exp $
#
# special R=0.15 IC jets:
iterativeCone15CaloJets = cms.EDProducer("IterativeConeJetProducer",
    #       using IconeJetParameters
    CaloJetParameters,
    alias = cms.untracked.string('IC15CaloJet'),
    coneRadius = cms.double(0.15),
    seedThreshold = cms.double(0.5),
    debugLevel = cms.untracked.int32(0)
)

recoJetsGR = cms.Sequence(iterativeCone15CaloJets+kt4CaloJets+kt6CaloJets+iterativeCone5CaloJets+sisCone5CaloJets+sisCone7CaloJets)

