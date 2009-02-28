import FWCore.ParameterSet.Config as cms
import copy

from RecoTauTag.RecoTau.PFRecoTauProducer_cfi import *


"""
        Defines producers and discriminants for the "ShrinkingCone" PFTau

        The sequence provided @ the end of the file,

                ProduceAndDiscriminateShrinkingConePFTaus 

        produces the shrinking cone PFTau and all its associated discriminants

        Signal/Iso cone parameters:
           SignalCone for tracks           - 5/ET in DR from lead object, min 0.07, max 0.15
           SignalCone for ECAL/HCAL        - 5/ET in DR from lead object, min 0.07, max 0.15
           Isolation cone (all types0      - 0.50 in DR from lead object
"""

ShrinkingConePFTauProducer = copy.deepcopy(pfRecoTauProducer)
# Lower pT thresholds to support tau neural classifier
ShrinkingConePFTauProducer.GammaCand_minPt              = cms.double(0.5)
ShrinkingConePFTauProducer.ChargedHadrCand_minPt        = cms.double(0.5);
#All cones use DR metric
#SignalCone parameters
ShrinkingConePFTauProducer.TrackerSignalConeSizeFormula = cms.string('5/ET'), ## **
ShrinkingConePFTauProducer.ECALSignalConeSizeFormula    = cms.string('5/ET'), ## **
ShrinkingConePFTauProducer.HCALSignalConeSizeFormula    = cms.string('5/ET'), ## **
#Isolation Cone parameters
ShrinkingConePFTauProducer.TrackerIsolConeSizeFormula   = cms.string('0.50'), ## **
ShrinkingConePFTauProducer.ECALIsolConeSizeFormula      = cms.string('0.50'), ## **
ShrinkingConePFTauProducer.HCALIsolConeSizeFormula      = cms.string('0.50'), ## **

ShrinkingConePFTauProducer.TrackerSignalConeSize_min = cms.double(0.07),
ShrinkingConePFTauProducer.TrackerSignalConeSize_max = cms.double(0.15),
ShrinkingConePFTauProducer.TrackerIsolConeSize_min   = cms.double(0.0),
ShrinkingConePFTauProducer.TrackerIsolConeSize_max   = cms.double(0.6),
ShrinkingConePFTauProducer.ECALSignalConeSize_min    = cms.double(0.07),
ShrinkingConePFTauProducer.ECALSignalConeSize_max    = cms.double(0.15),
ShrinkingConePFTauProducer.ECALIsolConeSize_min      = cms.double(0.0),
ShrinkingConePFTauProducer.ECALIsolConeSize_max      = cms.double(0.6),
ShrinkingConePFTauProducer.HCALSignalConeSize_min    = cms.double(0.07),
ShrinkingConePFTauProducer.HCALSignalConeSize_max    = cms.double(0.15),
ShrinkingConePFTauProducer.HCALIsolConeSize_min      = cms.double(0.0),
ShrinkingConePFTauProducer.HCALIsolConeSize_max      = cms.double(0.6),

# Get the decay mode reconstruction producer
from RecoTauTag.RecoTau.PFRecoTauDecayModeDeterminator_cfi                          import *
ShrinkingConePFTauDecayModeProducer               = copy.deepcopy(pfTauDecayMode)
ShrinkingConePFTauDecayModeProducer.PFTauProducer = 'ShrinkingConePFTauProducer'

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
ShrinkingConeDiscriminationByLeadingTrackFinding                          = copy.deepcopy(pfRecoTauDiscriminationByLeadingTrackFinding)
ShrinkingConeDiscriminationByLeadingTrackFinding.PFTauProducer            = 'ShrinkingConePFTauProducer'

ShrinkingConeDiscriminationByLeadingTrackPtCut                            = copy.deepcopy(pfRecoTauDiscriminationByLeadingTrackPtCut)
ShrinkingConeDiscriminationByLeadingTrackPtCut.PFTauProducer              = 'ShrinkingConePFTauProducer'

#copying Discriminator ByPionTrackPtCut
ShrinkingConeDiscriminationByLeadingPionPtCut                             = copy.deepcopy(pfRecoTauDiscriminationByLeadingPionPtCut)
ShrinkingConeDiscriminationByLeadingPionPtCut.PFTauProducer               = 'ShrinkingConePFTauProducer'

#copying the Discriminator by Isolation
ShrinkingConeDiscriminationByIsolation                                    = copy.deepcopy(pfRecoTauDiscriminationByIsolation)
ShrinkingConeDiscriminationByIsolation.PFTauProducer                      = 'ShrinkingConePFTauProducer'

ShrinkingConeDiscriminationByTrackIsolation                               = copy.deepcopy(pfRecoTauDiscriminationByTrackIsolation)
ShrinkingConeDiscriminationByTrackIsolation.PFTauProducer                 = 'ShrinkingConePFTauProducer'

ShrinkingConeDiscriminationByECALIsolation                                = copy.deepcopy(pfRecoTauDiscriminationByECALIsolation)
ShrinkingConeDiscriminationByECALIsolation.PFTauProducer                  = 'ShrinkingConePFTauProducer'

#copying the Discriminator by Isolation for leadingPion
ShrinkingConeDiscriminationByIsolationUsingLeadingPion                    = copy.deepcopy(pfRecoTauDiscriminationByIsolationUsingLeadingPion)
ShrinkingConeDiscriminationByIsolationUsingLeadingPion.PFTauProducer      = 'ShrinkingConePFTauProducer'

ShrinkingConeDiscriminationByTrackIsolationUsingLeadingPion               = copy.deepcopy(pfRecoTauDiscriminationByTrackIsolationUsingLeadingPion)
ShrinkingConeDiscriminationByTrackIsolationUsingLeadingPion.PFTauProducer = 'ShrinkingConePFTauProducer'

ShrinkingConeDiscriminationByECALIsolationUsingLeadingPion                = copy.deepcopy(pfRecoTauDiscriminationByECALIsolationUsingLeadingPion)
ShrinkingConeDiscriminationByECALIsolationUsingLeadingPion.PFTauProducer  = 'ShrinkingConePFTauProducer'

#copying discriminator against electrons and muons
ShrinkingConeDiscriminationAgainstElectron                                = copy.deepcopy(pfRecoTauDiscriminationAgainstElectron)
ShrinkingConeDiscriminationAgainstElectron.PFTauProducer                  = 'ShrinkingConePFTauProducer'

ShrinkingConeDiscriminationAgainstMuon                                    = copy.deepcopy(pfRecoTauDiscriminationAgainstMuon)
ShrinkingConeDiscriminationAgainstMuon.PFTauProducer                      = 'ShrinkingConePFTauProducer'

ProduceAndDiscriminateShrinkingConePFTaus = cms.Sequence(
      ShrinkingConePFTauProducer*
      ShrinkingConePFTauDecayModeProducer*
      ShrinkingConeDiscriminationByLeadingTrackFinding*
      ShrinkingConeDiscriminationByLeadingTrackPtCut*
      ShrinkingConeDiscriminationByLeadingPionPtCut*
      ShrinkingConeDiscriminationByIsolation*
      ShrinkingConeDiscriminationByTrackIsolation*
      ShrinkingConeDiscriminationByECALIsolation*
      ShrinkingConeDiscriminationByIsolationUsingLeadingPion*
      ShrinkingConeDiscriminationByTrackIsolationUsingLeadingPion*
      ShrinkingConeDiscriminationByECALIsolationUsingLeadingPion*
      ShrinkingConeDiscriminationAgainstElectron*
      ShrinkingConeDiscriminationAgainstMuon
      )

