import FWCore.ParameterSet.Config as cms
import copy

from RecoTauTag.RecoTau.PFRecoTauProducer_cfi import *

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
fixedConePFTauProducer = copy.deepcopy(pfRecoTauProducer)

fixedConePFTauProducer.LeadPFCand_minPt      = cms.double(5.0)  #cut on lead object (can be track, or gamma)

#Signal Cone parameters
fixedConePFTauProducer.TrackerSignalConeSizeFormula = cms.string('0.07') ## **
fixedConePFTauProducer.ECALSignalConeSizeFormula    = cms.string('0.07') ## **
fixedConePFTauProducer.HCALSignalConeSizeFormula    = cms.string('0.07') ## **
#Isolation Cone parameters
fixedConePFTauProducer.TrackerIsolConeSizeFormula   = cms.string('0.50') ## **
fixedConePFTauProducer.ECALIsolConeSizeFormula      = cms.string('0.50') ## **
fixedConePFTauProducer.HCALIsolConeSizeFormula      = cms.string('0.50') ## **

# Get the decay mode reconstruction producer
from RecoTauTag.RecoTau.PFRecoTauDecayModeDeterminator_cfi                          import *
fixedConePFTauDecayModeProducer               = copy.deepcopy(pfTauDecayMode)
fixedConePFTauDecayModeProducer.PFTauProducer = 'fixedConePFTauProducer'

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
fixedConeDiscriminationByLeadingTrackFinding                          = copy.deepcopy(pfRecoTauDiscriminationByLeadingTrackFinding)
fixedConeDiscriminationByLeadingTrackFinding.PFTauProducer            = 'fixedConePFTauProducer'

fixedConeDiscriminationByLeadingTrackPtCut                            = copy.deepcopy(pfRecoTauDiscriminationByLeadingTrackPtCut)
fixedConeDiscriminationByLeadingTrackPtCut.PFTauProducer              = 'fixedConePFTauProducer'

#copying Discriminator ByPionTrackPtCut
fixedConeDiscriminationByLeadingPionPtCut                             = copy.deepcopy(pfRecoTauDiscriminationByLeadingPionPtCut)
fixedConeDiscriminationByLeadingPionPtCut.PFTauProducer               = 'fixedConePFTauProducer'

#copying the Discriminator by Isolation
fixedConeDiscriminationByIsolation                                    = copy.deepcopy(pfRecoTauDiscriminationByIsolation)
fixedConeDiscriminationByIsolation.PFTauProducer                      = 'fixedConePFTauProducer'

fixedConeDiscriminationByTrackIsolation                               = copy.deepcopy(pfRecoTauDiscriminationByTrackIsolation)
fixedConeDiscriminationByTrackIsolation.PFTauProducer                 = 'fixedConePFTauProducer'

fixedConeDiscriminationByECALIsolation                                = copy.deepcopy(pfRecoTauDiscriminationByECALIsolation)
fixedConeDiscriminationByECALIsolation.PFTauProducer                  = 'fixedConePFTauProducer'

#copying the Discriminator by Isolation for leadingPion
fixedConeDiscriminationByIsolationUsingLeadingPion                    = copy.deepcopy(pfRecoTauDiscriminationByIsolationUsingLeadingPion)
fixedConeDiscriminationByIsolationUsingLeadingPion.PFTauProducer      = 'fixedConePFTauProducer'

fixedConeDiscriminationByTrackIsolationUsingLeadingPion               = copy.deepcopy(pfRecoTauDiscriminationByTrackIsolationUsingLeadingPion)
fixedConeDiscriminationByTrackIsolationUsingLeadingPion.PFTauProducer = 'fixedConePFTauProducer'

fixedConeDiscriminationByECALIsolationUsingLeadingPion                = copy.deepcopy(pfRecoTauDiscriminationByECALIsolationUsingLeadingPion)
fixedConeDiscriminationByECALIsolationUsingLeadingPion.PFTauProducer  = 'fixedConePFTauProducer'

#copying discriminator against electrons and muons
fixedConeDiscriminationAgainstElectron                                = copy.deepcopy(pfRecoTauDiscriminationAgainstElectron)
fixedConeDiscriminationAgainstElectron.PFTauProducer                  = 'fixedConePFTauProducer'

fixedConeDiscriminationAgainstMuon                                    = copy.deepcopy(pfRecoTauDiscriminationAgainstMuon)
fixedConeDiscriminationAgainstMuon.PFTauProducer                      = 'fixedConePFTauProducer'

produceAndDiscriminateFixedConePFTaus = cms.Sequence(
      fixedConePFTauProducer*
      fixedConePFTauDecayModeProducer*
      fixedConeDiscriminationByLeadingTrackFinding*
      fixedConeDiscriminationByLeadingTrackPtCut*
      fixedConeDiscriminationByLeadingPionPtCut*
      fixedConeDiscriminationByIsolation*
      fixedConeDiscriminationByTrackIsolation*
      fixedConeDiscriminationByECALIsolation*
      fixedConeDiscriminationByIsolationUsingLeadingPion*
      fixedConeDiscriminationByTrackIsolationUsingLeadingPion*
      fixedConeDiscriminationByECALIsolationUsingLeadingPion*
      fixedConeDiscriminationAgainstElectron*
      fixedConeDiscriminationAgainstMuon
      )

