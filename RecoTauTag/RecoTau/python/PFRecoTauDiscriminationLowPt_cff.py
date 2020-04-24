import FWCore.ParameterSet.Config as cms

from RecoTauTag.RecoTau.PFRecoTauDiscriminationByTrackIsolation_cfi  import pfRecoTauDiscriminationByTrackIsolation
from RecoTauTag.RecoTau.PFTauDiscriminatorLogicalAndProducer_cfi import PFTauDiscriminatorLogicalAndProducer

lowptpfTauDiscrByTrackIsolation = pfRecoTauDiscriminationByTrackIsolation.clone()
lowptpfTauDiscrByTrackIsolation.PFTauProducer  = cms.InputTag('pfLayer0Taus')
lowptpfTauDiscrByTrackIsolation.Prediscriminants.leadTrack.Producer = cms.InputTag('fixedConePFTauDiscriminationByLeadingTrackFinding')
lowptpfTauDiscrByTrackIsolation.ApplySumPtCut = cms.bool(True)
lowptpfTauDiscrByTrackIsolation.ApplyRelativeSumPtCut = cms.bool(False)
lowptpfTauDiscrByTrackIsolation.maximumSumPtCut                       = cms.double(1.0)
lowptpfTauDiscrByTrackIsolation.applyOccupancyCut                     = cms.bool(False)
lowptpfTauDiscrByTrackIsolation.qualityCuts.isolationQualityCuts.minTrackPt=cms.double(0.0)

lowptpfTauDiscrByRelTrackIsolation = pfRecoTauDiscriminationByTrackIsolation.clone()
lowptpfTauDiscrByRelTrackIsolation.PFTauProducer  = cms.InputTag('pfLayer0Taus')
lowptpfTauDiscrByRelTrackIsolation.Prediscriminants.leadTrack.Producer = cms.InputTag('fixedConePFTauDiscriminationByLeadingTrackFinding')
lowptpfTauDiscrByRelTrackIsolation.ApplySumPtCut = cms.bool(False)
lowptpfTauDiscrByRelTrackIsolation.ApplyRelativeSumPtCut = cms.bool(True)
lowptpfTauDiscrByRelTrackIsolation.maximumSumPtCut                       = cms.double(0.05)
lowptpfTauDiscrByRelTrackIsolation.applyOccupancyCut                     = cms.bool(True)
lowptpfTauDiscrByRelTrackIsolation.qualityCuts.isolationQualityCuts.minTrackPt=cms.double(0.0)

from RecoTauTag.RecoTau.PFRecoTauDiscriminationByCharge_cfi import pfRecoTauDiscriminationByCharge
pfRecoTauByCharge = pfRecoTauDiscriminationByCharge.clone()
pfRecoTauByCharge.PFTauProducer  = cms.InputTag('pfLayer0Taus')
pfRecoTauByCharge.ApplyOneOrThreeProngCut                    = cms.bool(True)


DiscrLowPtTau = PFTauDiscriminatorLogicalAndProducer.clone(
    PFTauProducer = cms.InputTag('pfLayer0Taus'),
    Prediscriminants = cms.PSet(
    BooleanOperator = cms.string("and"),
    isCharge = cms.PSet( Producer = cms.InputTag('pfRecoTauByCharge'),
                         cut      = cms.double(0.5)
                         ),
    lowptrel = cms.PSet( Producer = cms.InputTag('lowptpfTauDiscrByRelTrackIsolation'),
                         cut      = cms.double(0.5)
                         ),
    lowpt = cms.PSet( Producer = cms.InputTag('lowptpfTauDiscrByRelTrackIsolation'),
                      cut      = cms.double(0.5)
                      ),
    ),
    PassValue = cms.double(1.),
    FailValue = cms.double(0.)
    )


TauDiscrForLowPt = cms.Sequence(
    pfRecoTauByCharge+
    lowptpfTauDiscrByRelTrackIsolation+
    lowptpfTauDiscrByTrackIsolation+
    DiscrLowPtTau
    )
