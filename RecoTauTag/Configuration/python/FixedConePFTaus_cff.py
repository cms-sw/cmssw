import FWCore.ParameterSet.Config as cms
import copy

from RecoTauTag.RecoTau.RecoTauFixedConeProducer_cfi import fixedConeRecoTaus

"""
        Defines producers and discriminants for the "FixedCone" PFTau

        The sequence provided @ the end of the file,

                ProduceAndDiscriminateFixedConePFTaus

        produces the fixed cone PFTau and all its associated discriminants

        Signal/Iso cone parameters:
           SignalCone for tracks           - 0.07 in DR from lead object
           SignalCone for ECAL/HCAL        - 0.07 in DR from lead object
           Isolation cone (all types)      - 0.50 in DR from lead object

"""
fixedConePFTauProducer = copy.deepcopy(fixedConeRecoTaus)

# Get the decay mode reconstruction producer
from RecoTauTag.RecoTau.PFRecoTauDecayModeDeterminator_cfi                          import *
fixedConePFTauDecayModeProducer               = copy.deepcopy(pfTauDecayMode)
fixedConePFTauDecayModeProducer.PFTauProducer = 'fixedConePFTauProducer'

# Store the reco'd decay modes in a simple container
from RecoTauTag.RecoTau.PFRecoTauDecayModeIndexProducer_cfi                             import *
fixedConePFTauDecayModeIndexProducer                        = copy.deepcopy(pfTauDecayModeIndexProducer)
fixedConePFTauDecayModeIndexProducer.PFTauProducer          = cms.InputTag("fixedConePFTauProducer")
fixedConePFTauDecayModeIndexProducer.PFTauDecayModeProducer = cms.InputTag("fixedConePFTauDecayModeProducer")

# Define the discriminators for this tau
from RecoTauTag.RecoTau.PFRecoTauDiscriminationByIsolation_cfi                      import *
from RecoTauTag.RecoTau.PFRecoTauDiscriminationByLeadingTrackFinding_cfi            import *
from RecoTauTag.RecoTau.PFRecoTauDiscriminationByLeadingTrackPtCut_cfi              import *
from RecoTauTag.RecoTau.PFRecoTauDiscriminationByTrackIsolation_cfi                 import *
from RecoTauTag.RecoTau.PFRecoTauDiscriminationByECALIsolation_cfi                  import *
from RecoTauTag.RecoTau.PFRecoTauDiscriminationAgainstElectron_cfi                  import *
from RecoTauTag.RecoTau.PFRecoTauDiscriminationAgainstMuon_cfi                      import *
#Discriminators using leading Pion
from RecoTauTag.RecoTau.PFRecoTauDiscriminationByIsolationUsingLeadingPion_cfi      import *
from RecoTauTag.RecoTau.PFRecoTauDiscriminationByLeadingPionPtCut_cfi               import *
from RecoTauTag.RecoTau.PFRecoTauDiscriminationByTrackIsolationUsingLeadingPion_cfi import *
from RecoTauTag.RecoTau.PFRecoTauDiscriminationByECALIsolationUsingLeadingPion_cfi  import *

# Load helper functions to change the source of the discriminants
from RecoTauTag.RecoTau.TauDiscriminatorTools import *

#copying Discriminator ByLeadingTrack(finding and pt_cut)
fixedConePFTauDiscriminationByLeadingTrackFinding                          = copy.deepcopy(pfRecoTauDiscriminationByLeadingTrackFinding)
setTauSource(fixedConePFTauDiscriminationByLeadingTrackFinding, 'fixedConePFTauProducer')

fixedConePFTauDiscriminationByLeadingTrackPtCut                            = copy.deepcopy(pfRecoTauDiscriminationByLeadingTrackPtCut)
setTauSource(fixedConePFTauDiscriminationByLeadingTrackPtCut, 'fixedConePFTauProducer')

#copying Discriminator ByPionTrackPtCut
fixedConePFTauDiscriminationByLeadingPionPtCut                             = copy.deepcopy(pfRecoTauDiscriminationByLeadingPionPtCut)
setTauSource(fixedConePFTauDiscriminationByLeadingPionPtCut, 'fixedConePFTauProducer')

#copying the Discriminator by Isolation
fixedConePFTauDiscriminationByIsolation                                    = copy.deepcopy(pfRecoTauDiscriminationByIsolation)
setTauSource(fixedConePFTauDiscriminationByIsolation, 'fixedConePFTauProducer')

fixedConePFTauDiscriminationByTrackIsolation                               = copy.deepcopy(pfRecoTauDiscriminationByTrackIsolation)
setTauSource(fixedConePFTauDiscriminationByTrackIsolation, 'fixedConePFTauProducer')

fixedConePFTauDiscriminationByECALIsolation                                = copy.deepcopy(pfRecoTauDiscriminationByECALIsolation)
setTauSource(fixedConePFTauDiscriminationByECALIsolation, 'fixedConePFTauProducer')

#copying the Discriminator by Isolation for leadingPion
fixedConePFTauDiscriminationByIsolationUsingLeadingPion                    = copy.deepcopy(pfRecoTauDiscriminationByIsolationUsingLeadingPion)
setTauSource(fixedConePFTauDiscriminationByIsolationUsingLeadingPion, 'fixedConePFTauProducer')

fixedConePFTauDiscriminationByTrackIsolationUsingLeadingPion               = copy.deepcopy(pfRecoTauDiscriminationByTrackIsolationUsingLeadingPion)
setTauSource(fixedConePFTauDiscriminationByTrackIsolationUsingLeadingPion, 'fixedConePFTauProducer')

fixedConePFTauDiscriminationByECALIsolationUsingLeadingPion                = copy.deepcopy(pfRecoTauDiscriminationByECALIsolationUsingLeadingPion)
setTauSource(fixedConePFTauDiscriminationByECALIsolationUsingLeadingPion, 'fixedConePFTauProducer')

#copying discriminator against electrons and muons
fixedConePFTauDiscriminationAgainstElectron                                = copy.deepcopy(pfRecoTauDiscriminationAgainstElectron)
setTauSource(fixedConePFTauDiscriminationAgainstElectron, 'fixedConePFTauProducer')

fixedConePFTauDiscriminationAgainstMuon                                    = copy.deepcopy(pfRecoTauDiscriminationAgainstMuon)
setTauSource(fixedConePFTauDiscriminationAgainstMuon, 'fixedConePFTauProducer')

produceAndDiscriminateFixedConePFTaus = cms.Sequence(
      fixedConePFTauProducer*
      fixedConePFTauDiscriminationByLeadingTrackFinding*
      fixedConePFTauDiscriminationByLeadingTrackPtCut*
      fixedConePFTauDiscriminationByLeadingPionPtCut*
      fixedConePFTauDiscriminationByIsolation*
      fixedConePFTauDiscriminationByTrackIsolation*
      fixedConePFTauDiscriminationByECALIsolation*
      fixedConePFTauDiscriminationByIsolationUsingLeadingPion*
      fixedConePFTauDiscriminationByTrackIsolationUsingLeadingPion*
      fixedConePFTauDiscriminationByECALIsolationUsingLeadingPion*
      fixedConePFTauDiscriminationAgainstElectron*
      fixedConePFTauDiscriminationAgainstMuon
      )

