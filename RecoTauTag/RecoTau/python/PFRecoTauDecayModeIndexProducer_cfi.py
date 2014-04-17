import FWCore.ParameterSet.Config as cms
from RecoTauTag.RecoTau.TauDiscriminatorTools import noPrediscriminants

"""
        Produces a PFTauDiscriminator (Association<PFTau, float>) that maps
        PFTaus with the decay mode as determined by the PFTauDecayMode object.
        The mapping of the index to real decay modes is given in 
        DataFormats/TauReco/interface/PFTauDecayMode.h
"""

pfTauDecayModeIndexProducer = cms.EDProducer("PFRecoTauDecayModeIndexProducer",
      PFTauProducer = cms.InputTag("pfRecoTauProducer"),
      PFTauDecayModeProducer = cms.InputTag("pfRecoTauDecayModeProducer"),

      # This discriminator automatically handles cases where the tau is 
      # empty, or has no lead track.  No prediscriminants are needed.
      Prediscriminants = noPrediscriminants,
)
