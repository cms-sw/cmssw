import FWCore.ParameterSet.Config as cms


from PhysicsTools.PFCandProducer.pfMET_cfi  import *
from PhysicsTools.PFCandProducer.pfPileUp_cff  import *
from PhysicsTools.PFCandProducer.pfElectrons_cff import *
from PhysicsTools.PFCandProducer.pfMuons_cff import *
from PhysicsTools.PFCandProducer.pfJets_cff import *
from PhysicsTools.PFCandProducer.pfTaus_cff import *
from PhysicsTools.PFCandProducer.pfTopProjection_cfi import *

PF2PAT = cms.Sequence(
    pfMET +
    pfPileUpSequence + 
    pfElectronSequence +
    pfMuonSequence + 
    pfJetSequence + 
    pfTauSequence +  
    pfTopProjection
    )


