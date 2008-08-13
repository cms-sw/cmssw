import FWCore.ParameterSet.Config as cms

#
from TrackingTools.TransientTrack.TransientTrackBuilder_cfi import *
from RecoJets.JetAssociationProducers.ic5PFJetTracksAssociatorAtVertex_cfi import *
#PFTauTagInfo Producer
from RecoTauTag.RecoTau.PFRecoTauTagInfoProducer_cfi import *
from RecoTauTag.RecoTau.PFRecoTauProducer_cfi import *
from RecoTauTag.RecoTau.PFRecoTauDiscriminationByIsolation_cfi import *
import copy
from RecoTauTag.RecoTau.PFRecoTauProducer_cfi import *
#PFRecoTau, Higher efficiency
pfRecoTauProducerHighEfficiency = copy.deepcopy(pfRecoTauProducer)
import copy
from RecoTauTag.RecoTau.PFRecoTauDiscriminationByIsolation_cfi import *
pfRecoTauDiscriminationHighEfficiency = copy.deepcopy(pfRecoTauDiscriminationByIsolation)
PFTau = cms.Sequence(ic5PFJetTracksAssociatorAtVertex*pfRecoTauTagInfoProducer*pfRecoTauProducer*pfRecoTauProducerHighEfficiency*pfRecoTauDiscriminationByIsolation*pfRecoTauDiscriminationHighEfficiency)
pfRecoTauProducerHighEfficiency.TrackerSignalConeSizeFormula = '5.0/ET'
pfRecoTauProducerHighEfficiency.TrackerSignalConeSize_min = 0.07
pfRecoTauProducerHighEfficiency.TrackerSignalConeSize_max = 0.15
pfRecoTauProducerHighEfficiency.GammaCand_minPt = 1.5
pfRecoTauDiscriminationHighEfficiency.PFTauProducer = 'pfRecoTauProducerHighEfficiency'

