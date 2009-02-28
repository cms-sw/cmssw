import FWCore.ParameterSet.Config as cms
import copy

from RecoTauTag.RecoTau.PFRecoTauProducer_cfi import *

"""
        Defines producers and discriminants for the "FixedCone" PFTau

        The sequence provided @ the end of the file,

                ProduceAndDiscriminateFixedConePFTaus 

        produces the fixed cone PFTau and all its associated discriminants

        Signal/Iso cone parameters:
           SignalCone for tracks           - 0.15 in DR from lead object
           SignalCone for ECAL/HCAL        - 0.15 in DR from lead object
           Isolation cone (all types)      - 0.50 in DR from lead object
        
        
"""
FixedConePFTauProducer = copy.deepcopy(pfRecoTauProducer)

FixedConePFTauProducer.LeadPFCand_mintPt    = cms.double(5.0),  #cut on lead object (can be track, or gamma)

#Pt cuts applied to objects
FixedConePFTauProducer.ChargedHadrCand_minPt = cms.double(1.0),
FixedConePFTauProducer.GammaCand_minPt       = cms.double(1.5), 

#Signal Cone parameters
FixedConePFTauProducer.TrackerSignalConeSizeFormula = cms.string('0.15'), ## **
FixedConePFTauProducer.ECALSignalConeSizeFormula    = cms.string('0.15'), ## **
FixedConePFTauProducer.HCALSignalConeSizeFormula    = cms.string('0.15'), ## **
#Isolation Cone parameters
FixedConePFTauProducer.TrackerIsolConeSizeFormula   = cms.string('0.50'), ## **
FixedConePFTauProducer.ECALIsolConeSizeFormula      = cms.string('0.50'), ## **
FixedConePFTauProducer.HCALIsolConeSizeFormula      = cms.string('0.50'), ## **

# Get the decay mode reconstruction producer
from RecoTauTag.RecoTau.PFRecoTauDecayModeDeterminator_cfi                          import *
FixedConePFTauDecayModeProducer               = copy.deepcopy(pfTauDecayMode)
FixedConePFTauDecayModeProducer.PFTauProducer = 'FixedConePFTauProducer'


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
FixedConeDiscriminationByLeadingTrackFinding                          = copy.deepcopy(pfRecoTauDiscriminationByLeadingTrackFinding)
FixedConeDiscriminationByLeadingTrackFinding.PFTauProducer            = 'FixedConePFTauProducer'

FixedConeDiscriminationByLeadingTrackPtCut                            = copy.deepcopy(pfRecoTauDiscriminationByLeadingTrackPtCut)
FixedConeDiscriminationByLeadingTrackPtCut.PFTauProducer              = 'FixedConePFTauProducer'

#copying Discriminator ByPionTrackPtCut
FixedConeDiscriminationByLeadingPionPtCut                             = copy.deepcopy(pfRecoTauDiscriminationByLeadingPionPtCut)
FixedConeDiscriminationByLeadingPionPtCut.PFTauProducer               = 'FixedConePFTauProducer'

#copying the Discriminator by Isolation
FixedConeDiscriminationByIsolation                                    = copy.deepcopy(pfRecoTauDiscriminationByIsolation)
FixedConeDiscriminationByIsolation.PFTauProducer                      = 'FixedConePFTauProducer'

FixedConeDiscriminationByTrackIsolation                               = copy.deepcopy(pfRecoTauDiscriminationByTrackIsolation)
FixedConeDiscriminationByTrackIsolation.PFTauProducer                 = 'FixedConePFTauProducer'

FixedConeDiscriminationByECALIsolation                                = copy.deepcopy(pfRecoTauDiscriminationByECALIsolation)
FixedConeDiscriminationByECALIsolation.PFTauProducer                  = 'FixedConePFTauProducer'

#copying the Discriminator by Isolation for leadingPion
FixedConeDiscriminationByIsolationUsingLeadingPion                    = copy.deepcopy(pfRecoTauDiscriminationByIsolationUsingLeadingPion)
FixedConeDiscriminationByIsolationUsingLeadingPion.PFTauProducer      = 'FixedConePFTauProducer'

FixedConeDiscriminationByTrackIsolationUsingLeadingPion               = copy.deepcopy(pfRecoTauDiscriminationByTrackIsolationUsingLeadingPion)
FixedConeDiscriminationByTrackIsolationUsingLeadingPion.PFTauProducer = 'FixedConePFTauProducer'

FixedConeDiscriminationByECALIsolationUsingLeadingPion                = copy.deepcopy(pfRecoTauDiscriminationByECALIsolationUsingLeadingPion)
FixedConeDiscriminationByECALIsolationUsingLeadingPion.PFTauProducer  = 'FixedConePFTauProducer'

#copying discriminator against electrons and muons
FixedConeDiscriminationAgainstElectron                                = copy.deepcopy(pfRecoTauDiscriminationAgainstElectron)
FixedConeDiscriminationAgainstElectron.PFTauProducer                  = 'FixedConePFTauProducer'

FixedConeDiscriminationAgainstMuon                                    = copy.deepcopy(pfRecoTauDiscriminationAgainstMuon)
FixedConeDiscriminationAgainstMuon.PFTauProducer                      = 'FixedConePFTauProducer'

ProduceAndDiscriminateFixedConePFTaus = cms.Sequence(
      FixedConePFTauProducer*
      FixedConePFTauDecayModeProducer*
      FixedConeDiscriminationByLeadingTrackFinding*
      FixedConeDiscriminationByLeadingTrackPtCut*
      FixedConeDiscriminationByLeadingPionPtCut*
      FixedConeDiscriminationByIsolation*
      FixedConeDiscriminationByTrackIsolation*
      FixedConeDiscriminationByECALIsolation*
      FixedConeDiscriminationByIsolationUsingLeadingPion*
      FixedConeDiscriminationByTrackIsolationUsingLeadingPion*
      FixedConeDiscriminationByECALIsolationUsingLeadingPion*
      FixedConeDiscriminationAgainstElectron*
      FixedConeDiscriminationAgainstMuon
      )

