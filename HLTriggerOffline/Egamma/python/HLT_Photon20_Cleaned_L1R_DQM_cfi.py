import FWCore.ParameterSet.Config as cms

##########################################################
# See HLT Config Browser, for up-to-date HLT paths
#  http://cms-project-confdb-hltdev.web.cern.ch/cms-project-confdb-hltdev/browser/
#
# This config is for
#  HLT_Photon20_Cleaned_L1R
#    A single photon trigger, requiring at least one HLT photon with ET > 20 GeV.
#    More info: https://twiki.cern.ch/twiki/bin/view/CMS/TSG_03_V_09_8E29
#
# This path contains 5 steps:
#  HLT_Photon20_Cleaned_L1R = { HLTBeginSequence &
#                               hltL1sL1SingleEG8 &
#                               hltPrePhoton20CleanedL1R &
#                               HLTSinglePhoton20CleanL1NonIsolatedHLTNonIsoSequence &
#                               HLTEndSequence }
#
# The sequence in step 4 has 13 steps, and 4 filters (marked with *)
#  HLTSinglePhoton20CleanL1NonIsolatedHLTNonIsoSequence = {
#     HLTDoRegionalEgammaEcalSequence &
#     HLTL1IsolatedEcalClustersSequence &
#     HLTL1NonIsolatedEcalClustersSequence &
#     hltL1IsoRecoEcalCandidate &
#     hltL1NonIsoRecoEcalCandidate &
#    *hltL1NonIsoHLTNonIsoSinglePhotonEt20CleanedL1MatchFilterRegional &
#    *hltL1NonIsoHLTNonIsoSinglePhotonEt20CleanedEtFilter &
#     HLTEgammaR9ShapeSequence &
#    *hltL1NonIsoHLTNonIsoSinglePhotonEt20CleanedR9ShapeFilter &
#     HLTDoLocalHcalWithoutHOSequence &
#     hltL1IsolatedPhotonHcalIsol &
#     hltL1NonIsolatedPhotonHcalIsol &
#    *hltL1NonIsoHLTNonIsoSinglePhotonEt20CleanedHcalIsolFilter }
#
# The filters (*) above are what go into
#  the "HLTCollectionLabels" below.
##########################################################

HLT_Photon20_Cleaned_L1R_DQM = cms.EDAnalyzer("EmDQM",
    triggerobject = cms.InputTag("hltTriggerSummaryRAW","","HLT"),
    pdgGen = cms.int32(22),     
    genEtaAcc = cms.double(2.5),
    genEtAcc = cms.double(10.0),
    reqNum = cms.uint32(1),
    PtMax = cms.untracked.double(80.0),
    useHumanReadableHistTitles = cms.untracked.bool(False),

    # Filters from collections listed above
    filters = cms.VPSet(
        ##########################################################
        #  Initial Collection                                    #
        ##########################################################
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
            HLTCollectionLabels = cms.InputTag("hltL1sL1SingleEG8","","HLT"),
            IsoCollections = cms.VInputTag(cms.InputTag("none")),
            theHLTOutputTypes = cms.int32(-82),
            HLTCollectionHumanName = cms.untracked.string("Level 1")
        ),
        ##########################################################
        #  L1 Object Matching Filter                             #
        ##########################################################
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
            HLTCollectionLabels = cms.InputTag("hltL1NonIsoHLTNonIsoSinglePhotonEt20CleanedL1MatchFilterRegional","","HLT"),
            IsoCollections = cms.VInputTag(cms.InputTag("none")),
            theHLTOutputTypes = cms.int32(92),
            HLTCollectionHumanName = cms.untracked.string("L1 Match Filter")
        ),
        ##########################################################
        #  Et Filter                                             #
        ##########################################################
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
            HLTCollectionLabels = cms.InputTag("hltL1NonIsoHLTNonIsoSinglePhotonEt20CleanedEtFilter","","HLT"),
            IsoCollections = cms.VInputTag(cms.InputTag("none")),
            theHLTOutputTypes = cms.int32(92),
            HLTCollectionHumanName = cms.untracked.string("Et Filter")
        ),
        ##########################################################
        #   ECAL Isolation Filter                                #
        ##########################################################
#        cms.PSet(
#            PlotBounds = cms.vdouble(0.0, 10.0),
#            HLTCollectionLabels = cms.InputTag("hltL1NonIsoSinglePhotonEt10EcalIsolFilter","","HLT"),
#            IsoCollections = cms.VInputTag(cms.InputTag("hltL1IsolatedPhotonEcalIsol","","HLT"), cms.InputTag("hltL1NonIsolatedPhotonEcalIsol","","HLT")),
#            theHLTOutputTypes = cms.int32(92),
#            HLTCollectionHumanName = cms.untracked.string("Ecal Iso Filter")
#        ),
        ##########################################################
        #  R9 Filter                                             #
        ##########################################################
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
            HLTCollectionLabels = cms.InputTag("hltL1NonIsoHLTNonIsoSinglePhotonEt20CleanedR9ShapeFilter","","HLT"),
            IsoCollections = cms.VInputTag(cms.InputTag("hltL1IsoR9shape","","HLT"), cms.InputTag("hltL1NonIsoR9shape","","HLT")),
            theHLTOutputTypes = cms.int32(92),
            HLTCollectionHumanName = cms.untracked.string("R9 Filter")
        ),
        ##########################################################
        #  HCAL Isolation Filter                                 #
        ##########################################################
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 10.0),
            HLTCollectionLabels = cms.InputTag("hltL1NonIsoHLTNonIsoSinglePhotonEt20CleanedHcalIsolFilter","","HLT"),
            IsoCollections = cms.VInputTag(cms.InputTag("hltL1IsolatedPhotonHcalIsol","","HLT"), cms.InputTag("hltL1NonIsolatedPhotonHcalIsol","","HLT")),
            theHLTOutputTypes = cms.int32(81),
            HLTCollectionHumanName = cms.untracked.string("Hcal Iso Filter")
        )
        ##########################################################
        #  Track Isolation Filter                                #
        ##########################################################
#        cms.PSet(
#            PlotBounds = cms.vdouble(0.0, 10.0),
#            HLTCollectionLabels = cms.InputTag("hltL1NonIsoSinglePhotonEt10TrackIsolFilter","","HLT"),
#            IsoCollections = cms.VInputTag(cms.InputTag("hltL1IsoPhotonTrackIsol","","HLT"), cms.InputTag("hltL1NonIsoPhotonTrackIsol","","HLT")),
#            theHLTOutputTypes = cms.int32(81),
#            HLTCollectionHumanName = cms.untracked.string("Track Iso Filter")
#        )
    )
)



