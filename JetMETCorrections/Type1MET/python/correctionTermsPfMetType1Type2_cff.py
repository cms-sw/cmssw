import FWCore.ParameterSet.Config as cms

##____________________________________________________________________________||
from JetMETCorrections.Configuration.JetCorrectors_cff import *

##____________________________________________________________________________||
# select PFCandidates ("unclustered energy") not within jets
# for Type 2 MET correction
from CommonTools.ParticleFlow.TopProjectors.pfNoJet_cfi import pfNoJet as _pfNoJet
# the new TopProjectors now work with Ptrs
# a conversion is needed if objects are not available
# add them upfront of the sequence
pfJetsPtrForMetCorr = cms.EDProducer("PFJetFwdPtrProducer",
   src = cms.InputTag("ak4PFJets")
)
# this one is needed only if the input file doesn't have it
# solved automatically with unscheduled execution
from RecoParticleFlow.PFProducer.pfLinker_cff import particleFlowPtrs
# particleFlowPtrs = cms.EDProducer("PFCandidateFwdPtrProducer",
#    src = cms.InputTag("particleFlow")
# )
# FIXME: THIS IS A WASTE, BUT NOT CLEAR HOW TO FIX IT CLEANLY: the module
# downstream operates with View<reco::Candidate>, I wish one could read
# it from std::vector<PFCandidateFwdPtr> directly
pfCandsNotInJetsPtrForMetCorr = _pfNoJet.clone(
    topCollection = cms.InputTag('pfJetsPtrForMetCorr'),
    bottomCollection = cms.InputTag('particleFlowPtrs')
)
pfCandsNotInJetsForMetCorr = cms.EDProducer("PFCandidateFromFwdPtrProducer",
    src = cms.InputTag("pfCandsNotInJetsPtrForMetCorr")
)

##____________________________________________________________________________||
corrPfMetType1 = cms.EDProducer(
    "PFJetMETcorrInputProducer",
    src = cms.InputTag('ak4PFJetsCHS'),
    offsetCorrLabel = cms.InputTag("ak4PFCHSL1FastjetCorrector"),
    jetCorrLabel = cms.InputTag("ak4PFCHSL1FastL2L3Corrector"), #for MC
    jetCorrLabelRes = cms.InputTag("ak4PFCHSL1FastL2L3ResidualCorrector"), # for data, automatic switch
    jetCorrEtaMax = cms.double(9.9),
    type1JetPtThreshold = cms.double(15.0),
    skipEM = cms.bool(True),
    skipEMfractionThreshold = cms.double(0.90),
    skipMuons = cms.bool(True),
    skipMuonSelection = cms.string("isGlobalMuon | isStandAloneMuon")
)

##____________________________________________________________________________||
pfCandMETcorr = cms.EDProducer(
    "PFCandMETcorrInputProducer",
    src = cms.InputTag('pfCandsNotInJetsForMetCorr'),
    srcWeights = cms.InputTag('')
    )

##____________________________________________________________________________||
corrPfMetType2 = cms.EDProducer(
    "Type2CorrectionProducer",
    srcUnclEnergySums = cms.VInputTag(
        cms.InputTag('corrPfMetType1', 'type2'),
        cms.InputTag('corrPfMetType1', 'offset'),
        cms.InputTag('pfCandMETcorr')
    ),
    type2CorrFormula = cms.string("A"),
    type2CorrParameter = cms.PSet(
        A = cms.double(1.4)
        )
    )

##____________________________________________________________________________||
correctionTermsPfMetType1Type2Task = cms.Task(
    pfJetsPtrForMetCorr,
    particleFlowPtrs,
    pfCandsNotInJetsPtrForMetCorr,
    pfCandsNotInJetsForMetCorr,
    pfCandMETcorr,
    ak4PFCHSL1FastL2L3ResidualCorrectorTask, #Data full chain
    ak4PFCHSL1FastL2L3CorrectorTask, #MC last corrector, previous are already in the data chain
    corrPfMetType1,
    corrPfMetType2
    )

correctionTermsPfMetType1Type2 = cms.Sequence(correctionTermsPfMetType1Type2Task)
