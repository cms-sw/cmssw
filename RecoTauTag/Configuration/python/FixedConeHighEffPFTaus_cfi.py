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

#copying Discriminator ByLeadingTrack(finding and pt_cut)
fixedConeHighEffDiscriminationByLeadingTrackFinding                          = copy.deepcopy(pfRecoTauDiscriminationByLeadingTrackFinding)
fixedConeHighEffDiscriminationByLeadingTrackFinding.PFTauProducer            = 'fixedConeHighEffPFTauProducer'

fixedConeHighEffDiscriminationByLeadingTrackPtCut                            = copy.deepcopy(pfRecoTauDiscriminationByLeadingTrackPtCut)
fixedConeHighEffDiscriminationByLeadingTrackPtCut.PFTauProducer              = 'fixedConeHighEffPFTauProducer'

#copying Discriminator ByPionTrackPtCut
fixedConeHighEffDiscriminationByLeadingPionPtCut                             = copy.deepcopy(pfRecoTauDiscriminationByLeadingPionPtCut)
fixedConeHighEffDiscriminationByLeadingPionPtCut.PFTauProducer               = 'fixedConeHighEffPFTauProducer'

#copying the Discriminator by Isolation
fixedConeHighEffDiscriminationByIsolation                                    = copy.deepcopy(pfRecoTauDiscriminationByIsolation)
fixedConeHighEffDiscriminationByIsolation.PFTauProducer                      = 'fixedConeHighEffPFTauProducer'

fixedConeHighEffDiscriminationByTrackIsolation                               = copy.deepcopy(pfRecoTauDiscriminationByTrackIsolation)
fixedConeHighEffDiscriminationByTrackIsolation.PFTauProducer                 = 'fixedConeHighEffPFTauProducer'

fixedConeHighEffDiscriminationByECALIsolation                                = copy.deepcopy(pfRecoTauDiscriminationByECALIsolation)
fixedConeHighEffDiscriminationByECALIsolation.PFTauProducer                  = 'fixedConeHighEffPFTauProducer'

#copying the Discriminator by Isolation for leadingPion
fixedConeHighEffDiscriminationByIsolationUsingLeadingPion                    = copy.deepcopy(pfRecoTauDiscriminationByIsolationUsingLeadingPion)
fixedConeHighEffDiscriminationByIsolationUsingLeadingPion.PFTauProducer      = 'fixedConeHighEffPFTauProducer'

fixedConeHighEffDiscriminationByTrackIsolationUsingLeadingPion               = copy.deepcopy(pfRecoTauDiscriminationByTrackIsolationUsingLeadingPion)
fixedConeHighEffDiscriminationByTrackIsolationUsingLeadingPion.PFTauProducer = 'fixedConeHighEffPFTauProducer'

fixedConeHighEffDiscriminationByECALIsolationUsingLeadingPion                = copy.deepcopy(pfRecoTauDiscriminationByECALIsolationUsingLeadingPion)
fixedConeHighEffDiscriminationByECALIsolationUsingLeadingPion.PFTauProducer  = 'fixedConeHighEffPFTauProducer'

#copying discriminator against electrons and muons
fixedConeHighEffDiscriminationAgainstElectron                                = copy.deepcopy(pfRecoTauDiscriminationAgainstElectron)
fixedConeHighEffDiscriminationAgainstElectron.PFTauProducer                  = 'fixedConeHighEffPFTauProducer'

fixedConeHighEffDiscriminationAgainstMuon                                    = copy.deepcopy(pfRecoTauDiscriminationAgainstMuon)
fixedConeHighEffDiscriminationAgainstMuon.PFTauProducer                      = 'fixedConeHighEffPFTauProducer'

produceAndDiscriminateFixedConeHighEffPFTaus = cms.Sequence(
      fixedConeHighEffPFTauProducer*
      fixedConeHighEffPFTauDecayModeProducer*
      fixedConeHighEffDiscriminationByLeadingTrackFinding*
      fixedConeHighEffDiscriminationByLeadingTrackPtCut*
      fixedConeHighEffDiscriminationByLeadingPionPtCut*
      fixedConeHighEffDiscriminationByIsolation*
      fixedConeHighEffDiscriminationByTrackIsolation*
      fixedConeHighEffDiscriminationByECALIsolation*
      fixedConeHighEffDiscriminationByIsolationUsingLeadingPion*
      fixedConeHighEffDiscriminationByTrackIsolationUsingLeadingPion*
      fixedConeHighEffDiscriminationByECALIsolationUsingLeadingPion*
      fixedConeHighEffDiscriminationAgainstElectron*
      fixedConeHighEffDiscriminationAgainstMuon
      )

