import FWCore.ParameterSet.Config as cms

from PhysicsTools.PFCandProducer.pfAllMuons_cfi  import *
from PhysicsTools.PFCandProducer.pfMuonsPtGt5_cfi import *
#from PhysicsTools.PFCandProducer.pfMuons_cfi import *
from PhysicsTools.PFCandProducer.pfNoMuons_cfi import *
from PhysicsTools.PFCandProducer.pfMuonIsolation_cff import *

pfMuonSequence = cms.Sequence(
    pfAllMuons +
    pfMuonsPtGt5 +
    #pfMuons
    pfMuonIsol
    )




