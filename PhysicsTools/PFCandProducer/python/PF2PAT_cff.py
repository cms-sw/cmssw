import FWCore.ParameterSet.Config as cms

from PhysicsTools.PFCandProducer.genMetTrue_cff  import *

from PhysicsTools.PFCandProducer.pfMET_cfi  import *
from PhysicsTools.PFCandProducer.pfPileUp_cfi  import *
from PhysicsTools.PFCandProducer.pfElectrons_cff import *
from PhysicsTools.PFCandProducer.pfMuons_cff import *
from PhysicsTools.PFCandProducer.pfJets_cff import *
from PhysicsTools.PFCandProducer.pfTaus_cff import *

# sequential top projection cleaning
from PhysicsTools.PFCandProducer.TopProjectors.noPileUp_cfi import *
from PhysicsTools.PFCandProducer.sortByType_cff import *
from PhysicsTools.PFCandProducer.TopProjectors.noMuon_cfi import * 
from PhysicsTools.PFCandProducer.TopProjectors.noElectron_cfi import * 
from PhysicsTools.PFCandProducer.TopProjectors.noJet_cfi import *
from PhysicsTools.PFCandProducer.TopProjectors.noTau_cfi import *


PF2PAT = cms.Sequence(
    genMetTrueSequence + 
    pfMET +
    pfPileUp +
    noPileUp + 
    sortByTypeSequence +
    pfElectronSequence +
    pfMuonSequence + 
    noMuon +
#    noElectron + 
# when uncommenting, change the source of the jet clustering
    pfJetSequence +
    noJet + 
    pfTauSequence +
    noTau
    )


