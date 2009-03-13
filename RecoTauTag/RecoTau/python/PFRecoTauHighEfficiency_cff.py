import FWCore.ParameterSet.Config as cms
import copy

from RecoTauTag.RecoTau.PFRecoTauTagInfoProducer_cfi                                import *
from RecoTauTag.RecoTau.PFRecoTauProducer_cfi                                       import *
from RecoTauTag.RecoTau.PFRecoTauDiscriminationByIsolation_cfi                      import *
from RecoTauTag.RecoTau.PFRecoTauDiscriminationByIsolationUsingLeadingPion_cfi      import *
from RecoTauTag.RecoTau.PFRecoTauDiscriminationByLeadingTrackFinding_cfi            import *
from RecoTauTag.RecoTau.PFRecoTauDiscriminationByLeadingTrackPtCut_cfi              import *
from RecoTauTag.RecoTau.PFRecoTauDiscriminationByLeadingPionPtCut_cfi               import *
from RecoTauTag.RecoTau.PFRecoTauDiscriminationByTrackIsolation_cfi                 import *
from RecoTauTag.RecoTau.PFRecoTauDiscriminationByTrackIsolationUsingLeadingPion_cfi import *
from RecoTauTag.RecoTau.PFRecoTauDiscriminationByECALIsolation_cfi                  import *
from RecoTauTag.RecoTau.PFRecoTauDiscriminationByECALIsolationUsingLeadingPion_cfi  import *
from RecoTauTag.RecoTau.PFRecoTauDiscriminationAgainstElectron_cfi                  import *
from RecoTauTag.RecoTau.PFRecoTauDiscriminationAgainstMuon_cfi                      import *

#copying the PFTau producer and select the shrinkingCone
pfRecoTauProducerHighEfficiency                              = copy.deepcopy(pfRecoTauProducer)
pfRecoTauProducerHighEfficiency.TrackerSignalConeSizeFormula = '5.0/ET'
pfRecoTauProducerHighEfficiency.TrackerSignalConeSize_min    = 0.07
pfRecoTauProducerHighEfficiency.TrackerSignalConeSize_max    = 0.15

#copying Discriminator ByLeadingTrack(finding and pt_cut)
pfRecoTauDiscriminationByLeadingTrackFindingHighEfficiency                          = copy.deepcopy(pfRecoTauDiscriminationByLeadingTrackFinding)
pfRecoTauDiscriminationByLeadingTrackFindingHighEfficiency.PFTauProducer            = 'pfRecoTauProducerHighEfficiency'

pfRecoTauDiscriminationByLeadingTrackPtCutHighEfficiency                            = copy.deepcopy(pfRecoTauDiscriminationByLeadingTrackPtCut)
pfRecoTauDiscriminationByLeadingTrackPtCutHighEfficiency.PFTauProducer              = 'pfRecoTauProducerHighEfficiency'

#copying Discriminator ByPionTrackPtCut
pfRecoTauDiscriminationByLeadingPionPtCutHighEfficiency                             = copy.deepcopy(pfRecoTauDiscriminationByLeadingPionPtCut)
pfRecoTauDiscriminationByLeadingPionPtCutHighEfficiency.PFTauProducer               = 'pfRecoTauProducerHighEfficiency'

#copying the Discriminator by Isolation
pfRecoTauDiscriminationByIsolationHighEfficiency                                    = copy.deepcopy(pfRecoTauDiscriminationByIsolation)
pfRecoTauDiscriminationByIsolationHighEfficiency.PFTauProducer                      = 'pfRecoTauProducerHighEfficiency'

pfRecoTauDiscriminationByTrackIsolationHighEfficiency                               = copy.deepcopy(pfRecoTauDiscriminationByTrackIsolation)
pfRecoTauDiscriminationByTrackIsolationHighEfficiency.PFTauProducer                 = 'pfRecoTauProducerHighEfficiency'

pfRecoTauDiscriminationByECALIsolationHighEfficiency                                = copy.deepcopy(pfRecoTauDiscriminationByECALIsolation)
pfRecoTauDiscriminationByECALIsolationHighEfficiency.PFTauProducer                  = 'pfRecoTauProducerHighEfficiency'

#copying the Discriminator by Isolation for leadingPion
pfRecoTauDiscriminationByIsolationUsingLeadingPionHighEfficiency                    = copy.deepcopy(pfRecoTauDiscriminationByIsolationUsingLeadingPion)
pfRecoTauDiscriminationByIsolationUsingLeadingPionHighEfficiency.PFTauProducer      = 'pfRecoTauProducerHighEfficiency'

pfRecoTauDiscriminationByTrackIsolationUsingLeadingPionHighEfficiency               = copy.deepcopy(pfRecoTauDiscriminationByTrackIsolationUsingLeadingPion)
pfRecoTauDiscriminationByTrackIsolationUsingLeadingPionHighEfficiency.PFTauProducer = 'pfRecoTauProducerHighEfficiency'

pfRecoTauDiscriminationByECALIsolationUsingLeadingPionHighEfficiency                = copy.deepcopy(pfRecoTauDiscriminationByECALIsolationUsingLeadingPion)
pfRecoTauDiscriminationByECALIsolationUsingLeadingPionHighEfficiency.PFTauProducer  = 'pfRecoTauProducerHighEfficiency'

#copying discriminator against electrons and muons
pfRecoTauDiscriminationAgainstElectronHighEfficiency                                = copy.deepcopy(pfRecoTauDiscriminationAgainstElectron)
pfRecoTauDiscriminationAgainstElectronHighEfficiency.PFTauProducer                  = 'pfRecoTauProducerHighEfficiency'

pfRecoTauDiscriminationAgainstMuonHighEfficiency                                    = copy.deepcopy(pfRecoTauDiscriminationAgainstMuon)
pfRecoTauDiscriminationAgainstMuonHighEfficiency.PFTauProducer                      = 'pfRecoTauProducerHighEfficiency'


