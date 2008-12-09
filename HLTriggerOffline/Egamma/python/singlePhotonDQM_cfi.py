import FWCore.ParameterSet.Config as cms

##########################################################
# See HLT Config Browser, for up-to-date HLT paths
#  http://cms-project-confdb-hltdev.web.cern.ch/cms-project-confdb-hltdev/browser/
#
# Current Paths of interest:
#  HLT_IsoPhoton10_L1R
#  HLT_Photon15_L1R
#
#
# An example, that first path contains 5 steps:
# path HLT_IsoPhoton10_L1R = { HLTBeginSequence &
#                              hltL1sRelaxedSingleEgammaEt8 &
#                              hltPreIsoPhoton10L1R &
#                              HLTSinglePhotonEt10L1NonIsolatedSequence &
#                              HLTEndSequence }
#
# And the sequence named in step 4 can be expanded into its 20 steps,
# and notice it contains 5 filters (marked with *)
# sequence HLTSinglePhotonEt10L1NonIsolatedSequence = {
# 1    HLTDoRegionalEgammaEcalSequence &
# 2    HLTL1IsolatedEcalClustersSequence &
# 3    HLTL1NonIsolatedEcalClustersSequence &
# 4    hltL1IsoRecoEcalCandidate &
# 5    hltL1NonIsoRecoEcalCandidate &
# 6*   hltL1NonIsoSinglePhotonEt10L1MatchFilterRegional &
# 7*   hltL1NonIsoSinglePhotonEt10EtFilter &
# 8    hltL1IsolatedPhotonEcalIsol &
# 9    hltL1NonIsolatedPhotonEcalIsol &
#10*   hltL1NonIsoSinglePhotonEt10EcalIsolFilter &
#11    HLTDoLocalHcalWithoutHOSequence &
#12    hltL1IsolatedPhotonHcalIsol &
#13    hltL1NonIsolatedPhotonHcalIsol &
#14*   hltL1NonIsoSinglePhotonEt10HcalIsolFilter &
#15    HLTDoLocalTrackerSequence &
#16    HLTL1IsoEgammaRegionalRecoTrackerSequence &
#17    HLTL1NonIsoEgammaRegionalRecoTrackerSequence & h
#18    ltL1IsoPhotonTrackIsol &
#19    hltL1NonIsoPhotonTrackIsol &
#20*   hltL1NonIsoSinglePhotonEt10TrackIsolFilter }
#
# The filters (*) above are what go into
#  the "HLTCollectionLabels" below.
##########################################################

singlePhotonDQM = cms.EDFilter("EmDQM",
    pdgGen = cms.int32(22),     
    genEtaAcc = cms.double(2.5),
    genEtAcc = cms.double(10.0),
    reqNum = cms.uint32(1),
    PtMax = cms.untracked.double(200.0),
    useHumanReadableHistTitles = cms.untracked.bool(False),

    # Filters from collections listed above
    filters = cms.VPSet(
        ##########################################################
        #  Initial Collection                                    #
        ##########################################################
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
            HLTCollectionLabels = cms.InputTag("hltL1sRelaxedSingleEgammaEt8","","HLT"),
            IsoCollections = cms.VInputTag(cms.InputTag("none")),
            theHLTOutputTypes = cms.uint32(83),
            HLTCollectionHumanName = cms.untracked.string("Level 1")
        ),
        ##########################################################
        #  L1 Object Matching Filter                             #
        ##########################################################
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
            HLTCollectionLabels = cms.InputTag("hltL1NonIsoSinglePhotonEt10L1MatchFilterRegional","","HLT"),
            IsoCollections = cms.VInputTag(cms.InputTag("none")),
            theHLTOutputTypes = cms.uint32(100),
            HLTCollectionHumanName = cms.untracked.string("L1 Match Filter")
        ),
        ##########################################################
        #  Et Filter                                             #
        ##########################################################
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
            HLTCollectionLabels = cms.InputTag("hltL1NonIsoSinglePhotonEt10EtFilter","","HLT"),
            IsoCollections = cms.VInputTag(cms.InputTag("none")),
            theHLTOutputTypes = cms.uint32(100),
            HLTCollectionHumanName = cms.untracked.string("Et Filter")
        ),
        ##########################################################
        #   ECAL Isolation Filter                                #
        ##########################################################
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 10.0),
            HLTCollectionLabels = cms.InputTag("hltL1NonIsoSinglePhotonEt10EcalIsolFilter","","HLT"),
            IsoCollections = cms.VInputTag(cms.InputTag("hltL1IsolatedPhotonEcalIsol","","HLT"), cms.InputTag("hltL1NonIsolatedPhotonEcalIsol","","HLT")),
            theHLTOutputTypes = cms.uint32(100),
            HLTCollectionHumanName = cms.untracked.string("Ecal Iso Filter")
        ),
        ##########################################################
        #  HCAL Isolation Filter                                 #
        ##########################################################
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 10.0),
            HLTCollectionLabels = cms.InputTag("hltL1NonIsoSinglePhotonEt10HcalIsolFilter","","HLT"),
            IsoCollections = cms.VInputTag(cms.InputTag("hltL1IsolatedPhotonHcalIsol","","HLT"), cms.InputTag("hltL1NonIsolatedPhotonHcalIsol","","HLT")),
            theHLTOutputTypes = cms.uint32(100),
            HLTCollectionHumanName = cms.untracked.string("Hcal Iso Filter")
        ),
        ##########################################################
        #  Track Isolation Filter                                #
        ##########################################################
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 10.0),
            HLTCollectionLabels = cms.InputTag("hltL1NonIsoSinglePhotonEt10TrackIsolFilter","","HLT"),
            IsoCollections = cms.VInputTag(cms.InputTag("hltL1IsoPhotonTrackIsol","","HLT"), cms.InputTag("hltL1NonIsoPhotonTrackIsol","","HLT")),
            theHLTOutputTypes = cms.uint32(91),
            HLTCollectionHumanName = cms.untracked.string("Track Iso Filter")
        )
    )
)



