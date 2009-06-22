import FWCore.ParameterSet.Config as cms

from PhysicsTools.PFCandProducer.ParticleSelectors.pfAllMuons_cfi  import *
from PhysicsTools.PFCandProducer.ParticleSelectors.pfMuonsPtGt5_cfi import *
from PhysicsTools.PFCandProducer.pfMuonIsolation_cff import *

pfMuonSequence = cms.Sequence(
    pfAllMuons +
    pfMuonsPtGt5 +
    pfMuonIsolationSequence
    )




