import FWCore.ParameterSet.Config as cms

##########################################################
# See HLT Config Browser, for up-to-date HLT paths
#  http://cms-project-confdb-hltdev.web.cern.ch/cms-project-confdb-hltdev/browser/
#
# This config is for
#  HLT_Photon15_LooseEcalIso_L1R
#    A single photon trigger, requiring at least one HLT photon with ET > 15 GeV.
#    Loose ECAL isolation is required, with < 3 GeV or 10% of the photon energy in the isolation cone. 
#    More info:  https://twiki.cern.ch/twiki/bin/view/CMS/TSG_03_V_09_8E29
#
#
#  HLT_Photon15_LooseEcalIso_L1R = { HLTBeginSequence &
#                                    hltL1sRelaxedSingleEgammaEt8 &
#                                    hltPrePhoton15LEIL1R &
#                                    HLTSinglePhoton15L1NonIsolatedHLTLEISequence &
#                                    HLTEndSequence }
#
# The sequence in step 4 has 14 steps, and 4 filters (marked with *)
#  HLTSinglePhoton15L1NonIsolatedHLTLEISequence = {
#  1   HLTDoRegionalEgammaEcalSequence +
#  2   HLTL1IsolatedEcalClustersSequence +
#  3   HLTL1NonIsolatedEcalClustersSequence +
#  4   hltL1IsoRecoEcalCandidate +
#  5   hltL1NonIsoRecoEcalCandidate +
#  6*  hltL1NonIsoSinglePhotonEt15LEIL1MatchFilterRegional +
#  7*  hltL1NonIsoSinglePhotonEt15LEIEtFilter +
#  8   hltL1IsolatedPhotonEcalIsol +
#  9   hltL1NonIsolatedPhotonEcalIsol +
# 10*  hltL1NonIsoSinglePhotonEt15LEIEcalIsolFilter +
# 11   HLTDoLocalHcalWithoutHOSequence +
# 12   hltL1IsolatedPhotonHcalIsol +
# 13   hltL1NonIsolatedPhotonHcalIsol +
# 14*  hltL1NonIsoSinglePhotonEt15LEIHcalIsolFilter )
#
# The filters (*) above are what go into
#  the "HLTCollectionLabels" below.
##########################################################

HLT_Photon15_LooseEcalIso_L1R_DQM = cms.EDAnalyzer("EmDQM",
    triggerobject = cms.InputTag("hltTriggerSummaryRAW","","HLT"),                            
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
            HLTCollectionLabels = cms.InputTag("hltL1NonIsoSinglePhotonEt15LEIL1MatchFilterRegional","","HLT"),
            IsoCollections = cms.VInputTag(cms.InputTag("none")),
            theHLTOutputTypes = cms.int32(92),
            HLTCollectionHumanName = cms.untracked.string("L1 Match Filter")
        ),
        ##########################################################
        #  Et Filter                                             #
        ##########################################################
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
            HLTCollectionLabels = cms.InputTag("hltL1NonIsoSinglePhotonEt15LEIEtFilter","","HLT"),
            IsoCollections = cms.VInputTag(cms.InputTag("none")),
            theHLTOutputTypes = cms.int32(92),
            HLTCollectionHumanName = cms.untracked.string("Et Filter")
        ),
        ##########################################################
        #   ECAL Isolation Filter                                #
        ##########################################################
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 10.0),
            HLTCollectionLabels = cms.InputTag("hltL1NonIsoSinglePhotonEt15LEIEcalIsolFilter","","HLT"),
            IsoCollections = cms.VInputTag(cms.InputTag("hltL1IsolatedPhotonEcalIsol","","HLT"), cms.InputTag("hltL1NonIsolatedPhotonEcalIsol","","HLT")),
            theHLTOutputTypes = cms.int32(92),
            HLTCollectionHumanName = cms.untracked.string("Ecal Iso Filter")
        ),
        ##########################################################
        #  HCAL Isolation Filter                                 #
        ##########################################################
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 10.0),
            HLTCollectionLabels = cms.InputTag("hltL1NonIsoSinglePhotonEt15LEIHcalIsolFilter","","HLT"),
            IsoCollections = cms.VInputTag(cms.InputTag("hltL1IsolatedPhotonHcalIsol","","HLT"), cms.InputTag("hltL1NonIsolatedPhotonHcalIsol","","HLT")),
            theHLTOutputTypes = cms.int32(81),
            HLTCollectionHumanName = cms.untracked.string("Hcal Iso Filter")
        )
        ##########################################################
        #  Track Isolation Filter                                #
        ##########################################################
#        cms.PSet(
#            PlotBounds = cms.vdouble(0.0, 10.0),
#            HLTCollectionLabels = cms.InputTag("hltL1NonIsoSinglePhotonEt15HTITrackIsolFilter","","HLT"),
#            IsoCollections = cms.VInputTag(cms.InputTag("hltL1IsoPhotonHollowTrackIsol","","HLT"), cms.InputTag("hltL1NonIsoPhotonHollowTrackIsol","","HLT")),
#            theHLTOutputTypes = cms.int32(81),
#            HLTCollectionHumanName = cms.untracked.string("Track Iso Filter")
#        )
    )
)



