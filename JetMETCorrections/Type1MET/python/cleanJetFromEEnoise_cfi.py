import FWCore.ParameterSet.Config as cms


#______________________________________________________#
PFCandidateJetsWithEEnoise = cms.EDProducer(
    "BadPFCandidateJetsEEnoiseProducer",
    jetsrc                     = cms.InputTag("slimmedJets"),
    userawPt                   = cms.bool(True),
    PtThreshold                = cms.double(75.0),
    MinEtaThreshold            = cms.double(2.65),
    MaxEtaThreshold            = cms.double(3.139)
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
        cms.InputTag("PFCandidateJetsWithEEnoise"))
    )
#___________________________________________________________#
cleanPFCandidates = cms.EDProducer(
    "CandPtrProjector",
    src  = cms.InputTag("packedPFCandidates"),
    veto = cms.InputTag("superbad")
    )



#__________________________________________________________#


fullsuperbadSequence = cms.Sequence(PFCandidateJetsWithEEnoise+
                                    pfcandidateClustered +
                                    pfcandidateForUnclusteredUnc +
                                    badUnclustered +
                                    superbad +
                                    cleanPFCandidates
                                    )
