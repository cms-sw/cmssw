import FWCore.ParameterSet.Config as cms

# monitoring/validation for HLT_Ele17_SW_TighterEleIdIsol_L1R path
#
# originally copied from HLT_Ele15_SW_L1RDQM_cfi.py
#
#
# for the moment, this does not contain meaningful values for the IsoCollections
# parameters (which are used to produce 2D plots for the isolation variables)
#

# contents of HLT_Ele17_SW_TighterEleIdIsol_L1R_v2 (from the HLT MC menu, after l1 seed
# and prescale module):
#
# HLTDoRegionalEgammaEcalSequence
# HLTL1IsolatedEcalClustersSequence
# HLTL1NonIsolatedEcalClustersSequence
# hltL1IsoRecoEcalCandidate
# hltL1NonIsoRecoEcalCandidate
#     hltL1NonIsoHLTNonIsoSingleElectronEt17TighterEleIdIsolL1MatchFilterRegional  # cluster match to L1 seed
#     hltL1NonIsoHLTNonIsoSingleElectronEt17TighterEleIdIsolEtFilter               # Et filter
# hltL1IsoHLTClusterShape
# hltL1NonIsoHLTClusterShape
#     hltL1NonIsoHLTNonIsoSingleElectronEt17TighterEleIdIsolClusterShapeFilter     # cluster shape
# hltL1IsolatedPhotonEcalIsol
# hltL1NonIsolatedPhotonEcalIsol
#     hltL1NonIsoHLTNonIsoSingleElectronEt17TIghterEleIdIsolEcalIsolFilter         # ECAL isolation 
# HLTDoLocalHcalWithoutHOSequence
# hltL1IsolatedPhotonHcalForHE
# hltL1NonIsolatedPhotonHcalForHE
#     hltL1NonIsoHLTNonIsoSingleElectronEt17TighterEleIdIsolHEFilter               # H/E filter ??
# hltL1IsolatedPhotonHcalIsol
# hltL1NonIsolatedPhotonHcalIsol
#     hltL1NonIsoHLTNonIsoSingleElectronEt17TighterEleIdIsolHcalIsolFilter         # HCAL isolation
# HLTDoLocalPixelSequence
# HLTDoLocalStripSequence
# hltL1IsoStartUpElectronPixelSeeds
# hltL1NonIsoStartUpElectronPixelSeeds
#     hltL1NonIsoHLTNonIsoSingleElectronEt17TighterEleIdIsolPixelMatchFilter       # require a pixel match
# hltCkfL1IsoTrackCandidates
# hltCtfL1IsoWithMaterialTracks
# hltPixelMatchElectronsL1Iso
# hltCkfL1NonIsoTrackCandidates
# hltCtfL1NonIsoWithMaterialTracks
# hltPixelMatchElectronsL1NonIso
#     hltL1NonIsoHLTNonIsoSingleElectronEt17TighterEleIdIsolOneOEMinusOneOPFilter  # 1/E - 1/p requirement
# hltElectronL1IsoDetaDphi
# hltElectronL1NonIsoDetaDphi
#     hltL1NonIsoHLTNonIsoSingleElectronEt17TighterEleIdIsolDetaFilter             # delta eta cut
#     hltL1NonIsoHLTNonIsoSingleElectronEt17TighterEleIdIsolDphiFilter             # delta phi cut
# HLTL1IsoEgammaRegionalRecoTrackerSequence
# HLTL1NonIsoEgammaRegionalRecoTrackerSequence
# hltL1IsoElectronTrackIsol
# hltL1NonIsoElectronTrackIsol
#     hltL1NonIsoHLTNonIsoSingleElectronEt17TighterEleIdIsolTrackIsolFilter        # track isolation
#  

from HLTriggerOffline.Egamma.TriggerTypeDefs_cfi import *

hltProcName = "HLT2"

HLT_Ele17_SW_TighterEleIdIsol_L1RDQM = cms.EDAnalyzer("EmDQM",
    triggerobject = cms.InputTag("hltTriggerSummaryRAW","",hltProcName),                            
    genEtaAcc = cms.double(2.5),
    genEtAcc = cms.double(2.0),
    reqNum = cms.uint32(1),
    filters = cms.VPSet(
         #----------------------------------------
         # L1 seed
         #----------------------------------------
         cms.PSet(
             PlotBounds = cms.vdouble(0.0, 0.0),
             HLTCollectionLabels = cms.InputTag("hltL1sL1SingleEG8","",hltProcName),
             IsoCollections = cms.VInputTag(cms.InputTag("none")),
             theHLTOutputTypes = cms.int32(TriggerL1NoIsoEG)
         ), 
 
         #----------------------------------------
         # Match of cluster to L1 seed
         #----------------------------------------
         cms.PSet(
             PlotBounds = cms.vdouble(0.0, 0.0),
             HLTCollectionLabels = cms.InputTag("hltL1NonIsoHLTNonIsoSingleElectronEt17TighterEleIdIsolL1MatchFilterRegional","",hltProcName),
             IsoCollections = cms.VInputTag(cms.InputTag("none")),
             theHLTOutputTypes = cms.int32(TriggerCluster)
         ),
 
         #----------------------------------------
         # cluster shape
         #----------------------------------------
         cms.PSet(
             PlotBounds = cms.vdouble(0.0, 0.0),
             HLTCollectionLabels = cms.InputTag("hltL1NonIsoHLTNonIsoSingleElectronEt17TighterEleIdIsolClusterShapeFilter","",hltProcName),
             IsoCollections = cms.VInputTag(cms.InputTag("none")),
             theHLTOutputTypes = cms.int32(TriggerCluster)
         ),
 
         #----------------------------------------
         # ECAL isolation
         #----------------------------------------
         cms.PSet(
             PlotBounds = cms.vdouble(0.0, 0.0),
             HLTCollectionLabels = cms.InputTag("hltL1NonIsoHLTNonIsoSingleElectronEt17TIghterEleIdIsolEcalIsolFilter","",hltProcName),
             IsoCollections = cms.VInputTag(cms.InputTag("none")),
             theHLTOutputTypes = cms.int32(TriggerCluster)
         ),
 
         #----------------------------------------
         # H/E filter ??
         #----------------------------------------
         cms.PSet(
             PlotBounds = cms.vdouble(0.0, 0.0),
             HLTCollectionLabels = cms.InputTag("hltL1NonIsoHLTNonIsoSingleElectronEt17TighterEleIdIsolHEFilter","",hltProcName),
             IsoCollections = cms.VInputTag(cms.InputTag("none")),
             theHLTOutputTypes = cms.int32(TriggerCluster)
         ),
 
         #----------------------------------------
         # HCAL isolation
         #----------------------------------------
 
         cms.PSet(
             PlotBounds = cms.vdouble(0.0, 0.0),
             HLTCollectionLabels = cms.InputTag("hltL1NonIsoHLTNonIsoSingleElectronEt17TighterEleIdIsolHcalIsolFilter","",hltProcName),
             IsoCollections = cms.VInputTag(cms.InputTag("none")),
             theHLTOutputTypes = cms.int32(TriggerCluster)
         ),
 
         #----------------------------------------
         # Pixel match
         #----------------------------------------
 
         cms.PSet(
             PlotBounds = cms.vdouble(0.0, 0.0),
             HLTCollectionLabels = cms.InputTag("hltL1NonIsoHLTNonIsoSingleElectronEt17TighterEleIdIsolPixelMatchFilter","",hltProcName),
             IsoCollections = cms.VInputTag(cms.InputTag("none")),
             theHLTOutputTypes = cms.int32(TriggerCluster)
         ),
 
        #----------------------------------------
        # 1/E - 1/p filter
        #----------------------------------------

        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
            HLTCollectionLabels = cms.InputTag("hltL1NonIsoHLTNonIsoSingleElectronEt17TighterEleIdIsolOneOEMinusOneOPFilter","",hltProcName),
            IsoCollections = cms.VInputTag(cms.InputTag("none")),

            # TriggerCluster gives 0 objects found   
            # theHLTOutputTypes = cms.int32(TriggerCluster)

            # TriggerElectron gives
            # Event content inconsistent: TriggerEventWithRefs contains invalid Refsinvalid refs for: hltL1NonIsoHLTNonIsoSingleElectronEt17TighterEleIdIsolOneOEMinusOneOPFilter
            theHLTOutputTypes = cms.int32(TriggerElectron)
        ),


         #----------------------------------------
         # delta eta requirement
         #----------------------------------------
 
         cms.PSet(
             PlotBounds = cms.vdouble(0.0, 0.0),
             HLTCollectionLabels = cms.InputTag("hltL1NonIsoHLTNonIsoSingleElectronEt17TighterEleIdIsolDetaFilter","",hltProcName),
             IsoCollections = cms.VInputTag(cms.InputTag("none")),
             theHLTOutputTypes = cms.int32(TriggerElectron)
         ),
 
         #----------------------------------------
         # delta phi requirement
         #----------------------------------------
 
         cms.PSet(
             PlotBounds = cms.vdouble(0.0, 0.0),
             HLTCollectionLabels = cms.InputTag("hltL1NonIsoHLTNonIsoSingleElectronEt17TighterEleIdIsolDphiFilter","",hltProcName),
             IsoCollections = cms.VInputTag(cms.InputTag("none")),
             theHLTOutputTypes = cms.int32(TriggerElectron)
         ),
 
         #----------------------------------------
         # track isolation requirement
         #----------------------------------------
 
         cms.PSet(
             PlotBounds = cms.vdouble(0.0, 0.0),
             HLTCollectionLabels = cms.InputTag("hltL1NonIsoHLTNonIsoSingleElectronEt17TighterEleIdIsolTrackIsolFilter","",hltProcName),
             IsoCollections = cms.VInputTag(cms.InputTag("none")),
             theHLTOutputTypes = cms.int32(TriggerElectron)
         ),

      ),
    PtMax = cms.untracked.double(100.0),
    pdgGen = cms.int32(11)
)



