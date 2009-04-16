import FWCore.ParameterSet.Config as cms

"""
        Produces a PFTauDiscriminator (Association<PFTau, float>) that maps
        PFTaus with the decay mode as determined by the PFTauDecayMode object.
        The mapping of the index to real decay modes is given in 
        DataFormats/TauReco/interface/PFTauDecayMode.h
"""

pfTauDecayModeIndexProducer = cms.EDProducer("PFRecoTauDecayModeIndexProducer",
      PFTauProducer = cms.InputTag("pfRecoTauProducer"),
      PFTauDecayModeProducer = cms.InputTag("pfRecoTauDecayModeProducer")
)
