'''
       RecoPFTauTag_cff.py

       Contacts:  Evan Friis    (friis@physics.ucdavis.edu)
                  Simone Gennai (gennai@cern.ch)

       Use:  Include the sequence 'PFTau' in your path.

       Run the standard PFTau production squences.

       Produces:
          Name                                  Signal Cone        Iso Cone        Sequence
          -------------------------------------------------------------------------------------------------------------------
          shrinkingConePFTauProducer            DR = 5.0/ET        DR = 0.5        produceAndDiscriminateShrinkingConePFTaus
          fixedConePFTauProducer                DR = 0.07          DR = 0.5        produceAndDiscriminateFixedConePFTaus

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
from RecoJets.JetAssociationProducers.ic5PFJetTracksAssociatorAtVertex_cfi \
        import ic5PFJetTracksAssociatorAtVertex

# Switch to anti-kt 5 jets.
# Eventually this should be implemented in RecoJets.JetAssociationProducers
ak5PFJetTracksAssociatorAtVertex = ic5PFJetTracksAssociatorAtVertex.clone()
ak5PFJetTracksAssociatorAtVertex.jets = cms.InputTag("ak5PFJets")

# PFTauTagInfos are wrappers around jets and provide tau specific quality cuts
# Required for the production of PFTaus.
from RecoTauTag.RecoTau.PFRecoTauTagInfoProducer_cfi import *

# Ensure ak5PFJets are used
pfRecoTauTagInfoProducer.PFJetTracksAssociatorProducer = \
        cms.InputTag("ak5PFJetTracksAssociatorAtVertex")

# Get the standard PFTau production sequeneces
from RecoTauTag.Configuration.FixedConePFTaus_cfi import *
from RecoTauTag.Configuration.ShrinkingConePFTaus_cfi import *
from RecoTauTag.Configuration.HPSPFTaus_cfi import *

PFTau = cms.Sequence(
    ak5PFJetTracksAssociatorAtVertex *
    pfRecoTauTagInfoProducer *
    produceAndDiscriminateShrinkingConePFTaus +
    produceShrinkingConeDiscriminationByTauNeuralClassifier +
    produceAndDiscriminateFixedConePFTaus + 
    produceAndDiscriminateHPSPFTaus 
)
