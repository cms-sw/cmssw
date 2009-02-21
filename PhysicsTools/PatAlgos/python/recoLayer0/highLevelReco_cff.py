import FWCore.ParameterSet.Config as cms

# electron id
from PhysicsTools.PatAlgos.recoLayer0.electronId_cff import *

# Electron Isolation 
from PhysicsTools.PatAlgos.recoLayer0.electronIsolation_cff import *

# Muon Isolation 
from PhysicsTools.PatAlgos.recoLayer0.muonIsolation_cff import *

# Tau Isolation 
from PhysicsTools.PatAlgos.recoLayer0.tauIsolation_cff import *

# Photon Isolation and ID
from PhysicsTools.PatAlgos.recoLayer0.photonIsolation_cff import *
from PhysicsTools.PatAlgos.recoLayer0.photonId_cff import *

# jet-parton matching
from PhysicsTools.PatAlgos.recoLayer0.jetFlavourId_cff import *

# additional b-tagging
from PhysicsTools.PatAlgos.recoLayer0.bTagging_cff import *

# additional tau discriminators
from PhysicsTools.PatAlgos.recoLayer0.tauDiscriminators_cff import *

# Jet track association and jet charge
from PhysicsTools.PatAlgos.recoLayer0.jetTracksCharge_cff import *

# JetMET corrections
from PhysicsTools.PatAlgos.recoLayer0.jetMETCorrections_cff import *

# default  sequence for extra high-level reconstruction
patHighLevelReco_withoutPFTau = cms.Sequence(
    patElectronId *
    patPhotonId *
    patJetFlavourId *
    patLayer0ElectronIsolation *
    patLayer0PhotonIsolation *
    patLayer0MuonIsolation *
    patLayer0PFTauIsolation *
    patLayer0BTagging *
    patLayer0JetMETCorrections *
    patLayer0JetTracksCharge
)

patHighLevelReco = cms.Sequence(
    patHighLevelReco_withoutPFTau *
    patLayer0PFTauIsolation *    
    patPFTauDiscrimination
)

