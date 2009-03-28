import FWCore.ParameterSet.Config as cms
import copy

from RecoTauTag.RecoTau.PFRecoTauProducer_cfi import *

from RecoTauTag.TauTagTools.TauNeuralClassifiers_cfi import *

"""
        Defines producers and discriminants for the "shrinkingCone" PFTau

        The sequence provided @ the end of the file,

                produceAndDiscriminateShrinkingConePFTaus 

        produces the shrinking cone PFTau and all its associated discriminants

        Signal/Iso cone parameters:
           SignalCone for tracks           - 5/ET in DR from lead object, min 0.07, max 0.15
           SignalCone for ECAL/HCAL        - 5/ET in DR from lead object, min 0.07, max 0.15
           Isolation cone (all types0      - 0.50 in DR from lead object
"""

shrinkingConePFTauProducer = copy.deepcopy(pfRecoTauProducer)

shrinkingConePFTauProducer.LeadPFCand_minPt      = cms.double(5.0)  #cut on lead object (can be track, or gamma)

#All cones use DR metric
#SignalCone parameters
shrinkingConePFTauProducer.TrackerSignalConeSizeFormula = cms.string('5/ET') ## **
shrinkingConePFTauProducer.ECALSignalConeSizeFormula    = cms.string('5/ET') ## **
shrinkingConePFTauProducer.HCALSignalConeSizeFormula    = cms.string('5/ET') ## **
#Isolation Cone parameters
shrinkingConePFTauProducer.TrackerIsolConeSizeFormula   = cms.string('0.50') ## **
shrinkingConePFTauProducer.ECALIsolConeSizeFormula      = cms.string('0.50') ## **
shrinkingConePFTauProducer.HCALIsolConeSizeFormula      = cms.string('0.50') ## **

shrinkingConePFTauProducer.TrackerSignalConeSize_min = cms.double(0.07)
shrinkingConePFTauProducer.TrackerSignalConeSize_max = cms.double(0.15)
shrinkingConePFTauProducer.ECALSignalConeSize_min    = cms.double(0.07)
shrinkingConePFTauProducer.ECALSignalConeSize_max    = cms.double(0.15)
shrinkingConePFTauProducer.HCALSignalConeSize_min    = cms.double(0.07)
shrinkingConePFTauProducer.HCALSignalConeSize_max    = cms.double(0.15)

# Isolation cone sizes - note that since the iso cone size is fixed [0.5],
#  the only requirement for these quantities is that min < 0.5 < max.
shrinkingConePFTauProducer.TrackerIsolConeSize_min   = cms.double(0.0)
shrinkingConePFTauProducer.TrackerIsolConeSize_max   = cms.double(0.6)
shrinkingConePFTauProducer.ECALIsolConeSize_min      = cms.double(0.0)
shrinkingConePFTauProducer.ECALIsolConeSize_max      = cms.double(0.6)
shrinkingConePFTauProducer.HCALIsolConeSize_min      = cms.double(0.0)
shrinkingConePFTauProducer.HCALIsolConeSize_max      = cms.double(0.6)

# Get the decay mode reconstruction producer
from RecoTauTag.RecoTau.PFRecoTauDecayModeDeterminator_cfi                          import *
shrinkingConePFTauDecayModeProducer               = copy.deepcopy(pfTauDecayMode)
shrinkingConePFTauDecayModeProducer.PFTauProducer = 'shrinkingConePFTauProducer'

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
shrinkingConePFTauDiscriminationByLeadingTrackFinding                          = copy.deepcopy(pfRecoTauDiscriminationByLeadingTrackFinding)
shrinkingConePFTauDiscriminationByLeadingTrackFinding.PFTauProducer            = 'shrinkingConePFTauProducer'

shrinkingConePFTauDiscriminationByLeadingTrackPtCut                            = copy.deepcopy(pfRecoTauDiscriminationByLeadingTrackPtCut)
shrinkingConePFTauDiscriminationByLeadingTrackPtCut.PFTauProducer              = 'shrinkingConePFTauProducer'

#copying Discriminator ByPionTrackPtCut
shrinkingConePFTauDiscriminationByLeadingPionPtCut                             = copy.deepcopy(pfRecoTauDiscriminationByLeadingPionPtCut)
shrinkingConePFTauDiscriminationByLeadingPionPtCut.PFTauProducer               = 'shrinkingConePFTauProducer'

#copying the Discriminator by Isolation
shrinkingConePFTauDiscriminationByIsolation                                    = copy.deepcopy(pfRecoTauDiscriminationByIsolation)
shrinkingConePFTauDiscriminationByIsolation.PFTauProducer                      = 'shrinkingConePFTauProducer'

shrinkingConePFTauDiscriminationByTrackIsolation                               = copy.deepcopy(pfRecoTauDiscriminationByTrackIsolation)
shrinkingConePFTauDiscriminationByTrackIsolation.PFTauProducer                 = 'shrinkingConePFTauProducer'

shrinkingConePFTauDiscriminationByECALIsolation                                = copy.deepcopy(pfRecoTauDiscriminationByECALIsolation)
shrinkingConePFTauDiscriminationByECALIsolation.PFTauProducer                  = 'shrinkingConePFTauProducer'

#copying the Discriminator by Isolation for leadingPion
shrinkingConePFTauDiscriminationByIsolationUsingLeadingPion                    = copy.deepcopy(pfRecoTauDiscriminationByIsolationUsingLeadingPion)
shrinkingConePFTauDiscriminationByIsolationUsingLeadingPion.PFTauProducer      = 'shrinkingConePFTauProducer'

shrinkingConePFTauDiscriminationByTrackIsolationUsingLeadingPion               = copy.deepcopy(pfRecoTauDiscriminationByTrackIsolationUsingLeadingPion)
shrinkingConePFTauDiscriminationByTrackIsolationUsingLeadingPion.PFTauProducer = 'shrinkingConePFTauProducer'

shrinkingConePFTauDiscriminationByECALIsolationUsingLeadingPion                = copy.deepcopy(pfRecoTauDiscriminationByECALIsolationUsingLeadingPion)
shrinkingConePFTauDiscriminationByECALIsolationUsingLeadingPion.PFTauProducer  = 'shrinkingConePFTauProducer'

#copying discriminator against electrons and muons
shrinkingConePFTauDiscriminationAgainstElectron                                = copy.deepcopy(pfRecoTauDiscriminationAgainstElectron)
shrinkingConePFTauDiscriminationAgainstElectron.PFTauProducer                  = 'shrinkingConePFTauProducer'

shrinkingConePFTauDiscriminationAgainstMuon                                    = copy.deepcopy(pfRecoTauDiscriminationAgainstMuon)
shrinkingConePFTauDiscriminationAgainstMuon.PFTauProducer                      = 'shrinkingConePFTauProducer'

produceAndDiscriminateShrinkingConePFTaus = cms.Sequence(
      shrinkingConePFTauProducer*
      shrinkingConePFTauDecayModeProducer*
      shrinkingConePFTauDiscriminationByLeadingTrackFinding*
      shrinkingConePFTauDiscriminationByLeadingTrackPtCut*
      shrinkingConePFTauDiscriminationByLeadingPionPtCut*
      shrinkingConePFTauDiscriminationByIsolation*
      shrinkingConePFTauDiscriminationByTrackIsolation*
      shrinkingConePFTauDiscriminationByECALIsolation*
      shrinkingConePFTauDiscriminationByIsolationUsingLeadingPion*
      shrinkingConePFTauDiscriminationByTrackIsolationUsingLeadingPion*
      shrinkingConePFTauDiscriminationByECALIsolationUsingLeadingPion*
      shrinkingConePFTauDiscriminationAgainstElectron*
      shrinkingConePFTauDiscriminationAgainstMuon
      )

produceShrinkingConeDiscriminationByTauNeuralClassifier = cms.Sequence(
      shrinkingConePFTauDiscriminationByTaNC*
      shrinkingConePFTauDiscriminationByTaNCfrOnePercent*
      shrinkingConePFTauDiscriminationByTaNCfrHalfPercent*
      shrinkingConePFTauDiscriminationByTaNCfrQuarterPercent*
      shrinkingConePFTauDiscriminationByTaNCfrTenthPercent
      )


