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

# Jet projection:
pfcandidateNojets = cms.EDProducer("CandPtrProjector",
                                   src  = cms.InputTag("packedPFCandidates"),
                                   veto = cms.InputTag("slimmedJets")
                                   )
# Electron projection:
pfcandidateNojetsNoele = cms.EDProducer("CandPtrProjector",
                                        src  = cms.InputTag("pfcandidateNojets"),
                                        veto = cms.InputTag("slimmedElectrons")
                                        )
# Muon projection:
pfcandidateNojetsNoeleNomu = cms.EDProducer("CandPtrProjector",
                                            src  = cms.InputTag("pfcandidateNojetsNoele"),
                                            veto = cms.InputTag("slimmedMuons")
                                            )
# Tau projection:
pfcandidateNojetsNoeleNomuNotau = cms.EDProducer("CandPtrProjector",
                                                 src  = cms.InputTag("pfcandidateNojetsNoeleNomu"),
                                                 veto = cms.InputTag("slimmedTaus")
                                                 )
# Photon projection:
pfcandidateForUnclusteredUnc = cms.EDProducer("CandPtrProjector",
                                              src  = cms.InputTag("pfcandidateNojetsNoeleNomuNotau"),
                                              veto = cms.InputTag("slimmedPhotons")
                                              )


#__________________________________________________________________#
bad1 = cms.EDFilter(
    "CandPtrSelector",
    src = cms.InputTag("pfcandidateForUnclusteredUnc"),
    cut = cms.string("abs(eta) > 2.65 && abs(eta) < 3.139")
    )

#_________________________________________________________#
superbad = cms.EDProducer(
    "CandViewMerger",
    src = cms.VInputTag(
        cms.InputTag("bad1"),
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
                                    pfcandidateNojets +
                                    pfcandidateNojetsNoele +
                                    pfcandidateNojetsNoeleNomu +
                                    pfcandidateNojetsNoeleNomuNotau +
                                    pfcandidateForUnclusteredUnc +
                                    bad1 +
                                    superbad +
                                    cleanPFCandidates
                                    )
