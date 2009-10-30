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
           SignalCone for ECAL/HCAL        - 0.15 in DR from lead object
           Isolation cone (all types0      - 0.50 in DR from lead object
"""

shrinkingConePFTauProducer = copy.deepcopy(pfRecoTauProducer)

shrinkingConePFTauProducer.LeadPFCand_minPt      = cms.double(5.0)  #cut on lead object (can be track, or gamma)

#All cones use DR metric
#SignalCone parameters
shrinkingConePFTauProducer.TrackerSignalConeSizeFormula = cms.string('5/ET') ## **
shrinkingConePFTauProducer.ECALSignalConeSizeFormula    = cms.string('0.15') ## **
shrinkingConePFTauProducer.HCALSignalConeSizeFormula    = cms.string('0.15') ## **
#Isolation Cone parameters
shrinkingConePFTauProducer.TrackerIsolConeSizeFormula   = cms.string('0.50') ## **
shrinkingConePFTauProducer.ECALIsolConeSizeFormula      = cms.string('0.50') ## **
shrinkingConePFTauProducer.HCALIsolConeSizeFormula      = cms.string('0.50') ## **

shrinkingConePFTauProducer.TrackerSignalConeSize_min = cms.double(0.07)
shrinkingConePFTauProducer.TrackerSignalConeSize_max = cms.double(0.15)

# These are fixed at 0.15, so these values are meaningless (as long as min < 0.15 < max!)
shrinkingConePFTauProducer.ECALSignalConeSize_min    = cms.double(0.00)
shrinkingConePFTauProducer.ECALSignalConeSize_max    = cms.double(0.50)
shrinkingConePFTauProducer.HCALSignalConeSize_min    = cms.double(0.00)
shrinkingConePFTauProducer.HCALSignalConeSize_max    = cms.double(0.50)

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

# Store the reco'd decay modes in a simple container
from RecoTauTag.RecoTau.PFRecoTauDecayModeIndexProducer_cfi                             import *
shrinkingConePFTauDecayModeIndexProducer                        = copy.deepcopy(pfTauDecayModeIndexProducer)
shrinkingConePFTauDecayModeIndexProducer.PFTauProducer          = cms.InputTag("shrinkingConePFTauProducer")
shrinkingConePFTauDecayModeIndexProducer.PFTauDecayModeProducer = cms.InputTag("shrinkingConePFTauDecayModeProducer")


# Apply the CV tranformation to the TaNC output.  This module is not run in the default sequences, you must add it manually
#  For details of the transformation, see RecoTauTag/RecoTau/plugins/PFTauDecayModeCVTransformation.cc
#from RecoTauTag.TauTagTools.TancCVTransform_cfi                                     import *
#shrinkingConePFTauTancCVTransform = copy.deepcopy(TauCVTransformPrototype)
#shrinkingConePFTauTancCVTransform.PFTauDecayModeSrc            = cms.InputTag("shrinkingConePFTauDecayModeIndexProducer")
#shrinkingConePFTauTancCVTransform.PFTauDiscriminantToTransform = cms.InputTag('shrinkingConePFTauDiscriminationByTaNC')
#shrinkingConePFTauTancCVTransform.preDiscriminants             = cms.VInputTag("shrinkingConePFTauDiscriminationByLeadingPionPtCut")
#UpdateTransform(shrinkingConePFTauTancCVTransform, TaNC_DecayModeOccupancy)

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
shrinkingConePFTauDiscriminationByLeadingTrackFinding                          = copy.deepcopy(pfRecoTauDiscriminationByLeadingTrackFinding)
setTauSource(shrinkingConePFTauDiscriminationByLeadingTrackFinding, 'shrinkingConePFTauProducer')

shrinkingConePFTauDiscriminationByLeadingTrackPtCut                            = copy.deepcopy(pfRecoTauDiscriminationByLeadingTrackPtCut)
setTauSource(shrinkingConePFTauDiscriminationByLeadingTrackPtCut, 'shrinkingConePFTauProducer')

#copying Discriminator ByPionTrackPtCut
shrinkingConePFTauDiscriminationByLeadingPionPtCut                             = copy.deepcopy(pfRecoTauDiscriminationByLeadingPionPtCut)
setTauSource(shrinkingConePFTauDiscriminationByLeadingPionPtCut, 'shrinkingConePFTauProducer')

#copying the Discriminator by Isolation
shrinkingConePFTauDiscriminationByIsolation                                    = copy.deepcopy(pfRecoTauDiscriminationByIsolation)
setTauSource(shrinkingConePFTauDiscriminationByIsolation, 'shrinkingConePFTauProducer')

shrinkingConePFTauDiscriminationByTrackIsolation                               = copy.deepcopy(pfRecoTauDiscriminationByTrackIsolation)
setTauSource(shrinkingConePFTauDiscriminationByTrackIsolation, 'shrinkingConePFTauProducer')

shrinkingConePFTauDiscriminationByECALIsolation                                = copy.deepcopy(pfRecoTauDiscriminationByECALIsolation)
setTauSource(shrinkingConePFTauDiscriminationByECALIsolation, 'shrinkingConePFTauProducer')

#copying the Discriminator by Isolation for leadingPion
shrinkingConePFTauDiscriminationByIsolationUsingLeadingPion                    = copy.deepcopy(pfRecoTauDiscriminationByIsolationUsingLeadingPion)
setTauSource(shrinkingConePFTauDiscriminationByIsolationUsingLeadingPion, 'shrinkingConePFTauProducer')

shrinkingConePFTauDiscriminationByTrackIsolationUsingLeadingPion               = copy.deepcopy(pfRecoTauDiscriminationByTrackIsolationUsingLeadingPion)
setTauSource(shrinkingConePFTauDiscriminationByTrackIsolationUsingLeadingPion, 'shrinkingConePFTauProducer')

shrinkingConePFTauDiscriminationByECALIsolationUsingLeadingPion                = copy.deepcopy(pfRecoTauDiscriminationByECALIsolationUsingLeadingPion)
setTauSource(shrinkingConePFTauDiscriminationByECALIsolationUsingLeadingPion, 'shrinkingConePFTauProducer')

#copying discriminator against electrons and muons
shrinkingConePFTauDiscriminationAgainstElectron                                = copy.deepcopy(pfRecoTauDiscriminationAgainstElectron)
setTauSource(shrinkingConePFTauDiscriminationAgainstElectron, 'shrinkingConePFTauProducer')

shrinkingConePFTauDiscriminationAgainstMuon                                    = copy.deepcopy(pfRecoTauDiscriminationAgainstMuon)
setTauSource(shrinkingConePFTauDiscriminationAgainstMuon, 'shrinkingConePFTauProducer')

produceAndDiscriminateShrinkingConePFTaus = cms.Sequence(
      shrinkingConePFTauProducer*
      shrinkingConePFTauDecayModeProducer*
      shrinkingConePFTauDecayModeIndexProducer*
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


