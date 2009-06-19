import FWCore.ParameterSet.Config as cms

from PhysicsTools.PFCandProducer.pfIsol_cff import *
from PhysicsTools.PFCandProducer.genMetTrue_cff  import *
from PhysicsTools.PFCandProducer.pfMET_cfi  import *
from PhysicsTools.PFCandProducer.pfPileUp_cff  import *
from PhysicsTools.PFCandProducer.pfElectrons_cff import *
from PhysicsTools.PFCandProducer.pfMuons_cff import *
from PhysicsTools.PFCandProducer.pfJets_cff import *
from PhysicsTools.PFCandProducer.pfTaus_cff import *
#from PhysicsTools.PFCandProducer.pfTopProjection_cff import *

# sequential top projection cleaning
from PhysicsTools.PFCandProducer.pfNoPileUp_cff import *
from PhysicsTools.PFCandProducer.pfNoLepton_cff import *
from PhysicsTools.PFCandProducer.pfNoJet_cff import *
from PhysicsTools.PFCandProducer.pfJetsNoTau_cff import *


PF2PAT = cms.Sequence(
    genMetTrueSequence + 
    pfMET +
    pfPileUpSequence +
    pfNoPileUpSequence + 
    pfIsol +
    pfElectronSequence +
    pfMuonSequence +
    pfNoLeptonSequence + 
    pfJetSequence +
    pfNoJetSequence + 
    pfTauSequence +
    pfJetsNoTauSequence
    )


