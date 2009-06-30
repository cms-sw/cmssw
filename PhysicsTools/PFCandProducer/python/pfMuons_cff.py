import FWCore.ParameterSet.Config as cms

from PhysicsTools.PFCandProducer.pfPileUp_cff  import *
from PhysicsTools.PFCandProducer.ParticleSelectors.pfAllMuons_cfi  import *
from PhysicsTools.PFCandProducer.ParticleSelectors.pfMuonsPtGt5_cfi import *
from PhysicsTools.PFCandProducer.Isolation.muonIsolation_cff import *
from PhysicsTools.PFCandProducer.ParticleSelectors.isolatedMuons_cfi import *



pfMuonSequence = cms.Sequence(
    pfAllMuons +
    pfMuonsPtGt5 +
    # computing isolation variables:
    muonIsolationSequence +
    # selecting isolated electrons:
    isolatedMuons 
    )




