import FWCore.ParameterSet.Config as cms

##########################################################
# See HLT Config Browser, for up-to-date HLT paths
#  http://cms-project-confdb-hltdev.web.cern.ch/cms-project-confdb-hltdev/browser/
#
# Current Paths of interest:
#  HLT_IsoPhoton10_L1R
#  HLT_Photon15_L1R
#
# The path used in this config file is the 2nd one,
# path HLT_Photon15_L1R = { HLTBeginSequence &
#                           hltL1sRelaxedSingleEgammaEt10 &
#                           hltPrePhoton15L1R &
#                           HLTSinglePhoton15L1NonIsolatedHLTNonIsoSequence &
#                           HLTEndSequence }
#
# sequence HLTSinglePhoton15L1NonIsolatedHLTNonIsoSequence = {
# 1    HLTDoRegionalEgammaEcalSequence &
# 2    HLTL1IsolatedEcalClustersSequence &
# 3    HLTL1NonIsolatedEcalClustersSequence &
# 4    hltL1IsoRecoEcalCandidate &
# 5    hltL1NonIsoRecoEcalCandidate &
# 6*   hltL1NonIsoHLTNonIsoSinglePhotonEt15L1MatchFilterRegional &
# 7*   hltL1NonIsoHLTNonIsoSinglePhotonEt15EtFilter &
# 8    hltL1IsolatedPhotonEcalIsol &
# 9    hltL1NonIsolatedPhotonEcalIsol &
#10*   hltL1NonIsoHLTNonIsoSinglePhotonEt15EcalIsolFilter &
#11    HLTDoLocalHcalWithoutHOSequence &
#12    hltL1IsolatedPhotonHcalIsol &
#13    hltL1NonIsolatedPhotonHcalIsol &
#14*   hltL1NonIsoHLTNonIsoSinglePhotonEt15HcalIsolFilter &
#15    HLTDoLocalTrackerSequence &
#16    HLTL1IsoEgammaRegionalRecoTrackerSequence &
#17    HLTL1NonIsoEgammaRegionalRecoTrackerSequence &
#18    hltL1IsoPhotonTrackIsol &
#19    hltL1NonIsoPhotonTrackIsol &
#20*   hltL1NonIsoHLTNonIsoSinglePhotonEt15TrackIsolFilter }
#
# The filters (*) above are what go into
#  the "HLTCollectionLabels" below.
##########################################################


singlePhotonRelaxedDQM = cms.EDAnalyzer("EmDQM",
    pdgGen = cms.int32(22),
    genEtaAcc = cms.double(2.5),
    genEtAcc = cms.double(10.0),
    reqNum = cms.uint32(1),
    PtMax = cms.untracked.double(200.0),

    # Filters from collections listed above
    filters = cms.VPSet(
        ##########################################################
        #  Initial Collection                                    #
        ##########################################################
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
            HLTCollectionLabels = cms.InputTag("hltL1sRelaxedSingleEgammaEt10","","HLT"),
            IsoCollections = cms.VInputTag(cms.InputTag("none")),
            theHLTOutputTypes = cms.uint32(82)
        ),
        ##########################################################
        #  L1 Object Matching Filter                             #
        ##########################################################
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
            HLTCollectionLabels = cms.InputTag("hltL1NonIsoHLTNonIsoSinglePhotonEt15L1MatchFilterRegional","","HLT"),
            IsoCollections = cms.VInputTag(cms.InputTag("none")),
            theHLTOutputTypes = cms.uint32(100)
        ),
        ##########################################################
        #  Et Filter                                             #
        ##########################################################
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
            HLTCollectionLabels = cms.InputTag("hltL1NonIsoHLTNonIsoSinglePhotonEt15EtFilter","","HLT"),
            IsoCollections = cms.VInputTag(cms.InputTag("none")),
            theHLTOutputTypes = cms.uint32(100)
        ),
        ##########################################################
        #   ECAL Isolation Filter                                #
        ##########################################################
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 10.0),
            HLTCollectionLabels = cms.InputTag("hltL1NonIsoHLTNonIsoSinglePhotonEt15EcalIsolFilter","","HLT"),
            IsoCollections = cms.VInputTag(cms.InputTag("hltL1IsolatedPhotonEcalIsol","","HLT"), cms.InputTag("hltL1NonIsolatedPhotonEcalIsol","","HLT")),
            theHLTOutputTypes = cms.uint32(100)
        ),
        ##########################################################
        #  HCAL Isolation Filter                                 #
        ##########################################################
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 10.0),
            HLTCollectionLabels = cms.InputTag("hltL1NonIsoHLTNonIsoSinglePhotonEt15HcalIsolFilter","","HLT"),
            IsoCollections = cms.VInputTag(cms.InputTag("hltL1IsolatedPhotonHcalIsol","","HLT"), cms.InputTag("hltL1NonIsolatedPhotonHcalIsol","","HLT")),
            theHLTOutputTypes = cms.uint32(100)
        ),
        ##########################################################
        #  Track Isolation Filter                                #
        ##########################################################
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 10.0),
            HLTCollectionLabels = cms.InputTag("hltL1NonIsoHLTNonIsoSinglePhotonEt15TrackIsolFilter","","HLT"),
            IsoCollections = cms.VInputTag(cms.InputTag("hltL1IsoPhotonTrackIsol","","HLT"), cms.InputTag("hltL1NonIsoPhotonTrackIsol","","HLT")),
            theHLTOutputTypes = cms.uint32(91)
        )
    )
)



