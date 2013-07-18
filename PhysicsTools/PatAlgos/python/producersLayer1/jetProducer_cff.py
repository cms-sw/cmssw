import FWCore.ParameterSet.Config as cms

# prepare reco information
from PhysicsTools.PatAlgos.recoLayer0.jetTracksCharge_cff import *
from PhysicsTools.PatAlgos.recoLayer0.jetMETCorrections_cff import *
from PhysicsTools.PatAlgos.recoLayer0.bTagging_cff import *
#from PhysicsTools.PatAlgos.recoLayer0.jetID_cff import *

# add PAT specifics
from PhysicsTools.PatAlgos.mcMatchLayer0.jetFlavourId_cff import *
from PhysicsTools.PatAlgos.mcMatchLayer0.jetMatch_cfi import *

# produce object
from PhysicsTools.PatAlgos.producersLayer1.jetProducer_cfi import *

makePatJets = cms.Sequence(
    # reco pre-production
    patJetCorrections *
    patJetCharge *
   #secondaryVertexNegativeTagInfos *
   #simpleSecondaryVertexNegativeBJetTags *
    # pat specifics
    patJetPartonMatch *
    patJetGenJetMatch *
    patJetFlavourId *
    # object production
    patJets
    )
