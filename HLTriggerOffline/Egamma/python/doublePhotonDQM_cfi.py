import FWCore.ParameterSet.Config as cms

##########################################################
# See HLT Config Browser, for up-to-date HLT paths
#  http://cms-project-confdb-hltdev.web.cern.ch/cms-project-confdb-hltdev/browser/
#
# Current Paths of interest:
#  HLT_DoubleIsoPhoton20_L1R
#
# path HLT_DoubleIsoPhoton20_L1R = {
#  1    HLTBeginSequence &
#  2    hltL1sRelaxedDoubleEgamma &
#  3    HLTDoRegionalEgammaEcalSequence &
#  4    HLTL1IsolatedEcalClustersSequence &
#  5    HLTL1NonIsolatedEcalClustersSequence &
#  6    hltL1IsoRecoEcalCandidate &
#  7    hltL1NonIsoRecoEcalCandidate &
#  8*   hltL1NonIsoDoublePhotonL1MatchFilterRegional &
#  9*   hltL1NonIsoDoublePhotonEtFilter &
# 10    hltL1IsolatedPhotonEcalIsol &
# 11    hltL1NonIsolatedPhotonEcalIsol &
# 12*   hltL1NonIsoDoublePhotonEcalIsolFilter &
# 13    HLTDoLocalHcalWithoutHOSequence &
# 14    hltL1IsolatedPhotonHcalIsol &
# 15    hltL1NonIsolatedPhotonHcalIsol &
# 16*   hltL1NonIsoDoublePhotonHcalIsolFilter &
# 17    HLTDoLocalTrackerSequence &
# 18    HLTL1IsoEgammaRegionalRecoTrackerSequence &
# 19    HLTL1NonIsoEgammaRegionalRecoTrackerSequence &
# 20    hltL1IsoPhotonTrackIsol &
# 21    hltL1NonIsoPhotonTrackIsol &
# 22*   hltL1NonIsoDoublePhotonTrackIsolFilter &
# 23*   hltL1NonIsoDoublePhotonDoubleEtFilter &
# 24    hltDoublePhotonL1NonIsoPresc &
# 25    HLTEndSequence }
#
# The filters (*) above are what go into
#  the "HLTCollectionLabels" below.
##########################################################

doublePhotonDQM = cms.EDAnalyzer("EmDQM",
    PtMax = cms.untracked.double(200.0),
    pdgGen = cms.int32(22),
    genEtaAcc = cms.double(2.5),
    genEtAcc = cms.double(2.0),
    reqNum = cms.uint32(2),
                               
    # Filters from collections listed above
    filters = cms.VPSet(
        ##########################################################
        #  Initial Collection                                    #
        ##########################################################
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
#           HLTCollectionLabels = cms.InputTag("hltL1sDoubleEgamma","","HLT"),
            HLTCollectionLabels = cms.InputTag("hltL1sRelaxedDoubleEgamma","","HLT"),
            IsoCollections = cms.VInputTag(cms.InputTag("none")),
            theHLTOutputTypes = cms.uint32(83)
        ),
        ##########################################################
        #  L1 Object Matching Filter                             #
        ##########################################################
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
#           HLTCollectionLabels = cms.InputTag("hltL1IsoDoublePhotonL1MatchFilterRegional","","HLT"),
            HLTCollectionLabels = cms.InputTag("hltL1NonIsoDoublePhotonL1MatchFilterRegional","","HLT"),
            IsoCollections = cms.VInputTag(cms.InputTag("none")),
            theHLTOutputTypes = cms.uint32(100)
        ),
        ##########################################################
        #  Et Filter                                             #
        ##########################################################
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
#           HLTCollectionLabels = cms.InputTag("hltL1IsoDoublePhotonEtFilter","","HLT"),
            HLTCollectionLabels = cms.InputTag("hltL1NonIsoDoublePhotonEtFilter","","HLT"),
            IsoCollections = cms.VInputTag(cms.InputTag("none")),
            theHLTOutputTypes = cms.uint32(100)
        ),
        ##########################################################
        #   ECAL Isolation Filter                                #
        ##########################################################
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 10.0),
#           HLTCollectionLabels = cms.InputTag("hltL1IsoDoublePhotonEcalIsolFilter","","HLT"),
            HLTCollectionLabels = cms.InputTag("hltL1NonIsoDoublePhotonEcalIsolFilter","","HLT"),
#           IsoCollections = cms.VInputTag(cms.InputTag("hltL1IsolatedPhotonEcalIsol","","HLT")),
            IsoCollections = cms.VInputTag(cms.InputTag("hltL1IsolatedPhotonEcalIsol","","HLT"), cms.InputTag("hltL1NonIsolatedPhotonEcalIsol","","HLT")),
            theHLTOutputTypes = cms.uint32(100)
        ),
        ##########################################################
        #  HCAL Isolation Filter                                 #
        ##########################################################
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 10.0),
#           HLTCollectionLabels = cms.InputTag("hltL1IsoDoublePhotonHcalIsolFilter","","HLT"),
            HLTCollectionLabels = cms.InputTag("hltL1NonIsoDoublePhotonHcalIsolFilter","","HLT"),
#           IsoCollections = cms.VInputTag(cms.InputTag("hltL1IsolatedPhotonHcalIsol","","HLT")),
            IsoCollections = cms.VInputTag(cms.InputTag("hltL1IsolatedPhotonHcalIsol","","HLT"), cms.InputTag("hltL1NonIsolatedPhotonHcalIsol","","HLT")),
            theHLTOutputTypes = cms.uint32(100)
        ),
        ##########################################################
        #  Track Isolation Filter                                #
        ##########################################################
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 10.0),
#           HLTCollectionLabels = cms.InputTag("hltL1IsoDoublePhotonTrackIsolFilter","","HLT"),
            HLTCollectionLabels = cms.InputTag("hltL1NonIsoDoublePhotonTrackIsolFilter","","HLT"),
#           IsoCollections = cms.VInputTag(cms.InputTag("hltL1IsoPhotonTrackIsol","","HLT")),
            IsoCollections = cms.VInputTag(cms.InputTag("hltL1IsoPhotonTrackIsol"), cms.InputTag("hltL1NonIsoPhotonTrackIsol","","HLT")),
            theHLTOutputTypes = cms.uint32(91)
        ),
        ##########################################################
        #  DOUBLE Et Filter                                      #
        ##########################################################
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
            HLTCollectionLabels = cms.InputTag("hltL1NonIsoDoublePhotonDoubleEtFilter","","HLT"),
            IsoCollections = cms.VInputTag(cms.InputTag("none")),
            theHLTOutputTypes = cms.uint32(91)
        )
    )
)



