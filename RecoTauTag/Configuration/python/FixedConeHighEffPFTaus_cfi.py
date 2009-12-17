import FWCore.ParameterSet.Config as cms
import copy

from RecoTauTag.RecoTau.PFRecoTauProducer_cfi import *

"""
        Defines producers and discriminants for the "FixedCone High Efficiency" PFTau

        This uses a larger signal cone size (0.15 in Delta R) to improve tau caputre @ lower pt

        The sequence provided @ the end of the file,

                produceAndDiscriminateFixedConeHighEffPFTaus 

        produces the fixed cone PFTau and all its associated discriminants

        Signal/Iso cone parameters:
           SignalCone for tracks           - 0.15 in DR from lead object
           SignalCone for ECAL/HCAL        - 0.15 in DR from lead object
           Isolation cone (all types)      - 0.50 in DR from lead object
        
"""
fixedConeHighEffPFTauProducer = copy.deepcopy(pfRecoTauProducer)

fixedConeHighEffPFTauProducer.LeadPFCand_minPt      = cms.double(5.0)  #cut on lead object (can be track, or gamma)

#Signal Cone parameters
fixedConeHighEffPFTauProducer.TrackerSignalConeSizeFormula = cms.string('0.15') ## **
#Important: the four-vector energy for the PFTau is defined as the candidates
# within the maximum tracker signal cone size (TrackerSignalConeSize_max_).
# For fixed cone taus, this should be set to the fixed cone size.
fixedConeHighEffPFTauProducer.TrackerSignalConeSize_max    = cms.double(0.15) 

fixedConeHighEffPFTauProducer.ECALSignalConeSizeFormula    = cms.string('0.15') ## **
fixedConeHighEffPFTauProducer.HCALSignalConeSizeFormula    = cms.string('0.15') ## **
#Isolation Cone parameters
fixedConeHighEffPFTauProducer.TrackerIsolConeSizeFormula   = cms.string('0.50') ## **
fixedConeHighEffPFTauProducer.ECALIsolConeSizeFormula      = cms.string('0.50') ## **
fixedConeHighEffPFTauProducer.HCALIsolConeSizeFormula      = cms.string('0.50') ## **

# Get the decay mode reconstruction producer
from RecoTauTag.RecoTau.PFRecoTauDecayModeDeterminator_cfi                          import *
fixedConeHighEffPFTauDecayModeProducer               = copy.deepcopy(pfTauDecayMode)
fixedConeHighEffPFTauDecayModeProducer.PFTauProducer = 'fixedConeHighEffPFTauProducer'

# Store the reco'd decay modes in a simple container
from RecoTauTag.RecoTau.PFRecoTauDecayModeIndexProducer_cfi                             import *
fixedConeHighEffPFTauDecayModeIndexProducer                        = copy.deepcopy(pfTauDecayModeIndexProducer)
fixedConeHighEffPFTauDecayModeIndexProducer.PFTauProducer          = cms.InputTag("fixedConeHighEffPFTauProducer")
fixedConeHighEffPFTauDecayModeIndexProducer.PFTauDecayModeProducer = cms.InputTag("fixedConeHighEffPFTauDecayModeProducer")

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
fixedConeHighEffPFTauDiscriminationByLeadingTrackFinding                          = copy.deepcopy(pfRecoTauDiscriminationByLeadingTrackFinding)
setTauSource(fixedConeHighEffPFTauDiscriminationByLeadingTrackFinding, 'fixedConeHighEffPFTauProducer')

fixedConeHighEffPFTauDiscriminationByLeadingTrackPtCut                            = copy.deepcopy(pfRecoTauDiscriminationByLeadingTrackPtCut)
setTauSource(fixedConeHighEffPFTauDiscriminationByLeadingTrackPtCut, 'fixedConeHighEffPFTauProducer')

#copying Discriminator ByPionTrackPtCut
fixedConeHighEffPFTauDiscriminationByLeadingPionPtCut                             = copy.deepcopy(pfRecoTauDiscriminationByLeadingPionPtCut)
setTauSource(fixedConeHighEffPFTauDiscriminationByLeadingPionPtCut, 'fixedConeHighEffPFTauProducer')

#copying the Discriminator by Isolation
fixedConeHighEffPFTauDiscriminationByIsolation                                    = copy.deepcopy(pfRecoTauDiscriminationByIsolation)
setTauSource(fixedConeHighEffPFTauDiscriminationByIsolation, 'fixedConeHighEffPFTauProducer')

fixedConeHighEffPFTauDiscriminationByTrackIsolation                               = copy.deepcopy(pfRecoTauDiscriminationByTrackIsolation)
setTauSource(fixedConeHighEffPFTauDiscriminationByTrackIsolation, 'fixedConeHighEffPFTauProducer')

fixedConeHighEffPFTauDiscriminationByECALIsolation                                = copy.deepcopy(pfRecoTauDiscriminationByECALIsolation)
setTauSource(fixedConeHighEffPFTauDiscriminationByECALIsolation, 'fixedConeHighEffPFTauProducer')

#copying the Discriminator by Isolation for leadingPion
fixedConeHighEffPFTauDiscriminationByIsolationUsingLeadingPion                    = copy.deepcopy(pfRecoTauDiscriminationByIsolationUsingLeadingPion)
setTauSource(fixedConeHighEffPFTauDiscriminationByIsolationUsingLeadingPion, 'fixedConeHighEffPFTauProducer')

fixedConeHighEffPFTauDiscriminationByTrackIsolationUsingLeadingPion               = copy.deepcopy(pfRecoTauDiscriminationByTrackIsolationUsingLeadingPion)
setTauSource(fixedConeHighEffPFTauDiscriminationByTrackIsolationUsingLeadingPion, 'fixedConeHighEffPFTauProducer')

fixedConeHighEffPFTauDiscriminationByECALIsolationUsingLeadingPion                = copy.deepcopy(pfRecoTauDiscriminationByECALIsolationUsingLeadingPion)
setTauSource(fixedConeHighEffPFTauDiscriminationByECALIsolationUsingLeadingPion, 'fixedConeHighEffPFTauProducer')

#copying discriminator against electrons and muons
fixedConeHighEffPFTauDiscriminationAgainstElectron                                = copy.deepcopy(pfRecoTauDiscriminationAgainstElectron)
setTauSource(fixedConeHighEffPFTauDiscriminationAgainstElectron, 'fixedConeHighEffPFTauProducer')

fixedConeHighEffPFTauDiscriminationAgainstMuon                                    = copy.deepcopy(pfRecoTauDiscriminationAgainstMuon)
setTauSource(fixedConeHighEffPFTauDiscriminationAgainstMuon, 'fixedConeHighEffPFTauProducer')

produceAndDiscriminateFixedConeHighEffPFTaus = cms.Sequence(
      fixedConeHighEffPFTauProducer*
      fixedConeHighEffPFTauDiscriminationByLeadingTrackFinding*
      fixedConeHighEffPFTauDiscriminationByLeadingTrackPtCut*
      fixedConeHighEffPFTauDiscriminationByLeadingPionPtCut*
      fixedConeHighEffPFTauDiscriminationByIsolation*
      fixedConeHighEffPFTauDiscriminationByTrackIsolation*
      fixedConeHighEffPFTauDiscriminationByECALIsolation*
      fixedConeHighEffPFTauDiscriminationByIsolationUsingLeadingPion*
      fixedConeHighEffPFTauDiscriminationByTrackIsolationUsingLeadingPion*
      fixedConeHighEffPFTauDiscriminationByECALIsolationUsingLeadingPion*
      fixedConeHighEffPFTauDiscriminationAgainstElectron*
      fixedConeHighEffPFTauDiscriminationAgainstMuon
      )

