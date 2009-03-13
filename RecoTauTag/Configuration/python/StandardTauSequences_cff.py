'''
       StandardTauSequences_cff

       Contacts:  Evan Friis    (friis@physics.ucdavis.edu)
                  Simone Gennai (gennai@cern.ch)

       Use:  Include the sequence 'PFTaus' in your path.

       Run the standard PFTau production squences.

       Produces:
          Name                                  Signal Cone             Iso Cone
          --------------------------------------------------------------------------
          shrinkingConePFTauProducer            DR = 5.0/ET             DR = 0.5
          fixedConePFTauProducer                DR = 0.07               DR = 0.5
          fixedConeHighEffPFTauProducer         DR = 0.15               DR = 0.5

       A leading pion (charged or neutral) requirement of 5.0 GeV is applied in all cases.
       The PFTauDecayMode is produced for each, and contains additional information
       about the decay mode of the tau.

       A number of PFTauDiscriminators are automatically produced for each PFTau type.
         DiscriminationByLeadingTrackFinding
         DiscriminationByLeadingTrackPtCut
         DiscriminationByLeadingPionPtCut
         DiscriminationByIsolation
         DiscriminationByTrackIsolation
         DiscriminationByECALIsolation
         DiscriminationByIsolationUsingLeadingPion
         DiscriminationByTrackIsolationUsingLeadingPion
         DiscriminationByECALIsolationUsingLeadingPion
         DiscriminationAgainstElectron
         DiscriminationAgainstMuon
       See the relevant cfi in RecoTauTag/RecoTau/python for the discriminator parameters.
'''

import FWCore.ParameterSet.Config as cms

#Necessary for building PFTauTagInfos
from TrackingTools.TransientTrack.TransientTrackBuilder_cfi import *
from RecoJets.JetAssociationProducers.ic5PFJetTracksAssociatorAtVertex_cfi import *

# PFTauTagInfos are wrappers around jets and provide tau specific quality cuts
# Required for the production of PFTaus.
from RecoTauTag.RecoTau.PFRecoTauTagInfoProducer_cfi import *

# Get the standard PFTau production sequeneces
from RecoTauTag.Configuration.FixedConeHighEffPFTaus_cfi import *
from RecoTauTag.Configuration.FixedConePFTaus_cfi        import *
from RecoTauTag.Configuration.ShrinkingConePFTaus_cfi    import *

PFTaus = cms.Sequence(
      ic5PFJetTracksAssociatorAtVertex *
      pfRecoTauTagInfoProducer *
      produceAndDiscriminateShrinkingConePFTaus +
      produceAndDiscriminateFixedConeHighEffPFTaus + 
      produceAndDiscriminateFixedConePFTaus
      )
