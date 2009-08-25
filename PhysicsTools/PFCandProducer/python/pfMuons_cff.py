import FWCore.ParameterSet.Config as cms

from PhysicsTools.PFCandProducer.ParticleSelectors.pfAllMuons_cfi  import *
from PhysicsTools.PFCandProducer.ParticleSelectors.pfMuonsPtGt5_cfi import *
from PhysicsTools.PFCandProducer.Isolation.pfMuonIsolation_cff import *
from PhysicsTools.PFCandProducer.Isolation.pfIsolatedMuons_cfi import *



pfMuonSequence = cms.Sequence(
    pfAllMuons +
    pfMuonsPtGt5 +
    # computing isolation variables:
    pfMuonIsolationSequence +
    # selecting isolated electrons:
    pfIsolatedMuons 
    )




