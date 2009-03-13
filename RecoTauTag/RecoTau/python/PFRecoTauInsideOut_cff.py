import FWCore.ParameterSet.Config as cms
import copy

from RecoTauTag.RecoTau.PFRecoTauTagInfoProducer_cfi import *
from RecoTauTag.RecoTau.PFRecoTauProducer_cfi import *
from RecoTauTag.RecoTau.PFRecoTauDiscriminationByIsolation_cfi import *
from RecoTauTag.RecoTau.PFRecoTauDiscriminationByIsolationUsingLeadingPion_cfi import *
from RecoTauTag.RecoTau.PFRecoTauDiscriminationByLeadingTrackFinding_cfi import *
from RecoTauTag.RecoTau.PFRecoTauDiscriminationByLeadingTrackPtCut_cfi import *
from RecoTauTag.RecoTau.PFRecoTauDiscriminationByLeadingPionPtCut_cfi import *
from RecoTauTag.RecoTau.PFRecoTauDiscriminationByTrackIsolation_cfi import *
from RecoTauTag.RecoTau.PFRecoTauDiscriminationByTrackIsolationUsingLeadingPion_cfi import *
from RecoTauTag.RecoTau.PFRecoTauDiscriminationByECALIsolation_cfi import *
from RecoTauTag.RecoTau.PFRecoTauDiscriminationByECALIsolationUsingLeadingPion_cfi import *
from RecoTauTag.RecoTau.PFRecoTauDiscriminationAgainstElectron_cfi import *
from RecoTauTag.RecoTau.PFRecoTauDiscriminationAgainstMuon_cfi import *

#copying the PFTau producer and select the shrinkingCone
pfRecoTauProducerInsideOut = copy.deepcopy(pfRecoTauProducer)
pfRecoTauProducerInsideOut.PFTauTagInfoProducer = cms.InputTag('pfRecoTauTagInfoProducerInsideOut')
pfRecoTauProducerInsideOut.TrackerSignalConeSizeFormula = 'JetOpeningDR'
pfRecoTauProducerInsideOut.TrackerSignalConeSize_min    = 0.02
pfRecoTauProducerInsideOut.TrackerSignalConeSize_max    = 0.20

pfRecoTauProducerInsideOut.ECALSignalConeSizeFormula    = pfRecoTauProducerInsideOut.TrackerSignalConeSizeFormula
pfRecoTauProducerInsideOut.ECALSignalConeSize_min       = pfRecoTauProducerInsideOut.TrackerSignalConeSize_min
pfRecoTauProducerInsideOut.ECALSignalConeSize_max       = pfRecoTauProducerInsideOut.TrackerSignalConeSize_max

pfRecoTauProducerInsideOut.HCALSignalConeSizeFormula    = pfRecoTauProducerInsideOut.TrackerSignalConeSizeFormula
pfRecoTauProducerInsideOut.HCALSignalConeSize_min       = pfRecoTauProducerInsideOut.TrackerSignalConeSize_min
pfRecoTauProducerInsideOut.HCALSignalConeSize_max       = pfRecoTauProducerInsideOut.TrackerSignalConeSize_max

#copying Discriminator ByLeadingTrack(finding and pt_cut)
pfRecoTauDiscriminationByLeadingTrackFindingInsideOut                          = copy.deepcopy(pfRecoTauDiscriminationByLeadingTrackFinding)
pfRecoTauDiscriminationByLeadingTrackFindingInsideOut.PFTauProducer            = 'pfRecoTauProducerInsideOut'

pfRecoTauDiscriminationByLeadingTrackPtCutInsideOut                            = copy.deepcopy(pfRecoTauDiscriminationByLeadingTrackPtCut)
pfRecoTauDiscriminationByLeadingTrackPtCutInsideOut.PFTauProducer              = 'pfRecoTauProducerInsideOut'

#copying Discriminator ByPionTrackPtCut
pfRecoTauDiscriminationByLeadingPionPtCutInsideOut                             = copy.deepcopy(pfRecoTauDiscriminationByLeadingPionPtCut)
pfRecoTauDiscriminationByLeadingPionPtCutInsideOut.PFTauProducer               = 'pfRecoTauProducerInsideOut'

#copying the Discriminator by Isolation
pfRecoTauDiscriminationByIsolationInsideOut                                    = copy.deepcopy(pfRecoTauDiscriminationByIsolation)
pfRecoTauDiscriminationByIsolationInsideOut.PFTauProducer                      = 'pfRecoTauProducerInsideOut'

pfRecoTauDiscriminationByTrackIsolationInsideOut                               = copy.deepcopy(pfRecoTauDiscriminationByTrackIsolation)
pfRecoTauDiscriminationByTrackIsolationInsideOut.PFTauProducer                 = 'pfRecoTauProducerInsideOut'

pfRecoTauDiscriminationByECALIsolationInsideOut                                = copy.deepcopy(pfRecoTauDiscriminationByECALIsolation)
pfRecoTauDiscriminationByECALIsolationInsideOut.PFTauProducer                  = 'pfRecoTauProducerInsideOut'

#copying the Discriminator by Isolation for leadingPion
pfRecoTauDiscriminationByIsolationUsingLeadingPionInsideOut                    = copy.deepcopy(pfRecoTauDiscriminationByIsolationUsingLeadingPion)
pfRecoTauDiscriminationByIsolationUsingLeadingPionInsideOut.PFTauProducer      = 'pfRecoTauProducerInsideOut'

pfRecoTauDiscriminationByTrackIsolationUsingLeadingPionInsideOut               = copy.deepcopy(pfRecoTauDiscriminationByTrackIsolationUsingLeadingPion)
pfRecoTauDiscriminationByTrackIsolationUsingLeadingPionInsideOut.PFTauProducer = 'pfRecoTauProducerInsideOut'

pfRecoTauDiscriminationByECALIsolationUsingLeadingPionInsideOut                = copy.deepcopy(pfRecoTauDiscriminationByECALIsolationUsingLeadingPion)
pfRecoTauDiscriminationByECALIsolationUsingLeadingPionInsideOut.PFTauProducer  = 'pfRecoTauProducerInsideOut'

#copying discriminator against electrons and muons
pfRecoTauDiscriminationAgainstElectronInsideOut                                = copy.deepcopy(pfRecoTauDiscriminationAgainstElectron)
pfRecoTauDiscriminationAgainstElectronInsideOut.PFTauProducer                  = 'pfRecoTauProducerInsideOut'

pfRecoTauDiscriminationAgainstMuonInsideOut                                    = copy.deepcopy(pfRecoTauDiscriminationAgainstMuon)
pfRecoTauDiscriminationAgainstMuonInsideOut.PFTauProducer                      = 'pfRecoTauProducerInsideOut'


