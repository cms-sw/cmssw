import FWCore.ParameterSet.Config as cms

from PhysicsTools.PFCandProducer.GeneratorTools.genMetTrue_cff  import *

from PhysicsTools.PFCandProducer.pfMET_cfi  import *
from PhysicsTools.PFCandProducer.pfNoPileUp_cff  import *
from PhysicsTools.PFCandProducer.pfElectrons_cff import *
from PhysicsTools.PFCandProducer.pfMuons_cff import *
from PhysicsTools.PFCandProducer.pfJets_cff import *
from PhysicsTools.PFCandProducer.pfTaus_cff import *

# sequential top projection cleaning
from PhysicsTools.PFCandProducer.ParticleSelectors.pfSortByType_cff import *
from PhysicsTools.PFCandProducer.TopProjectors.pfNoMuon_cfi import * 
from PhysicsTools.PFCandProducer.TopProjectors.pfNoElectron_cfi import * 
from PhysicsTools.PFCandProducer.TopProjectors.pfNoJet_cfi import *
from PhysicsTools.PFCandProducer.TopProjectors.pfNoTau_cfi import *

# generator tools
from PhysicsTools.PFCandProducer.GeneratorTools.sortGenParticles_cff import *


PF2PAT = cms.Sequence(
    pfMET +
    pfNoPileUpSequence + 
    # pfSortByTypeSequence +
    pfAllNeutralHadrons+
    pfAllChargedHadrons+
    pfAllPhotons+
    pfAllMuons + 
    pfMuonSequence + 
    pfNoMuon +
    pfAllElectrons +
    pfElectronSequence +
    pfNoElectron + 
# when uncommenting, change the source of the jet clustering
    pfJetSequence +
    pfNoJet + 
    pfTauSequence +
    pfNoTau
    )


genForPF2PAT = cms.Sequence(
    genMetTrueSequence + 
    sortGenParticlesSequence
    )
