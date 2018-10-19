import FWCore.ParameterSet.Config as cms


#______________________________________________________#
pfCandidateJetsWithEEnoise = cms.EDProducer(
    "BadPFCandidateJetsEEnoiseProducer",
    jetsrc                     = cms.InputTag("slimmedJets"),
    userawPt                   = cms.bool(True),
    ptThreshold                = cms.double(50.0),
    minEtaThreshold            = cms.double(2.65),
    maxEtaThreshold            = cms.double(3.139)
    )


#_______________________________________________________#
# Construct the Unclustered PF Candidates

pfcandidateClustered = cms.EDProducer(
    "CandViewMerger",
    src = cms.VInputTag(
        cms.InputTag("slimmedJets"),
        cms.InputTag("slimmedElectrons"),
        cms.InputTag("slimmedMuons"),
        cms.InputTag("slimmedTaus"),
        cms.InputTag("slimmedPhotons"),
    )
)

pfcandidateForUnclusteredUnc = cms.EDProducer("CandPtrProjector",
    src  = cms.InputTag("packedPFCandidates"),
    veto = cms.InputTag("pfcandidateClustered"),
)


#__________________________________________________________________#
badUnclustered = cms.EDFilter("CandPtrSelector",
    src = cms.InputTag("pfcandidateForUnclusteredUnc"),
    cut = cms.string("abs(eta) > 2.65 && abs(eta) < 3.139")
)

#_________________________________________________________#
superbad = cms.EDProducer(
    "CandViewMerger",
    src = cms.VInputTag(
        cms.InputTag("badUnclustered"),
        cms.InputTag("pfCandidateJetsWithEEnoise"))
    )
#___________________________________________________________#
cleanPFCandidates = cms.EDProducer(
    "CandPtrProjector",
    src  = cms.InputTag("packedPFCandidates"),
    veto = cms.InputTag("superbad")
    )



#__________________________________________________________#


fullsuperbadSequence = cms.Sequence(pfCandidateJetsWithEEnoise+
                                    pfcandidateClustered +
                                    pfcandidateForUnclusteredUnc +
                                    badUnclustered +
                                    superbad +
                                    cleanPFCandidates
                                    )
