import FWCore.ParameterSet.Config as cms

# Set the HLT paths
import HLTrigger.HLTfilters.hltHighLevel_cfi
ALCARECOSiStripCalSmallBiasScanHLT = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone(
    andOr = True, ## choose logical OR between Triggerbits
    eventSetupPathsKey = 'SiStripCalSmallBiasScan',
    throw = False # tolerate triggers stated above, but not available
    )

# Select only events where tracker had HV on (according to DCS bit information)
# AND respective partition is in the run (according to FED information)
import CalibTracker.SiStripCommon.SiStripDCSFilter_cfi
DCSStatusForSiStripCalSmallBiasScan = CalibTracker.SiStripCommon.SiStripDCSFilter_cfi.siStripDCSFilter.clone()

from RecoVertex.BeamSpotProducer.BeamSpot_cff import *
from RecoTracker.IterativeTracking.InitialStep_cff import *
from RecoTracker.Configuration.RecoTrackerP5_cff import *
from RecoTracker.TrackProducer.TrackRefitter_cfi import *

################################################################################################
#TRACK REFITTER
################################################################################################
ALCARECOSiStripCalSmallBiasScanTracksRefit = TrackRefitter.clone(src = cms.InputTag("generalTracks"),
                                                                 NavigationSchool = cms.string("")
                                                                 )

################################################################################################
#TRACK FILTER
################################################################################################
import Calibration.TkAlCaRecoProducers.CalibrationTrackSelectorFromDetIdList_cfi as TrackSelectorFromDetIdList
ALCARECOSiStripCalSmallBiasScanSelectedTracks = TrackSelectorFromDetIdList.CalibrationTrackSelectorFromDetIdList.clone(Input= cms.InputTag("ALCARECOSiStripCalSmallBiasScanTracksRefit"),
                                                                                                                       selections=cms.VPSet(
        cms.PSet(detSelection = cms.uint32(1), detLabel = cms.string("TIB - 1.2.2.1")  ,selection=cms.untracked.vstring("0x1FFFFFFF-0x16005865")),
        cms.PSet(detSelection = cms.uint32(2), detLabel = cms.string("TIB - 1.2.2.1")  ,selection=cms.untracked.vstring("0x1FFFFFFF-0x16005866")),
        cms.PSet(detSelection = cms.uint32(3), detLabel = cms.string("TIB - 1.2.2.1")  ,selection=cms.untracked.vstring("0x1FFFFFFF-0x16005869")),
        cms.PSet(detSelection = cms.uint32(4), detLabel = cms.string("TIB - 1.2.2.1")  ,selection=cms.untracked.vstring("0x1FFFFFFF-0x1600586a")),
        cms.PSet(detSelection = cms.uint32(5), detLabel = cms.string("TIB - 1.2.2.1")  ,selection=cms.untracked.vstring("0x1FFFFFFF-0x1600586d")),
        cms.PSet(detSelection = cms.uint32(6), detLabel = cms.string("TIB - 1.2.2.1")  ,selection=cms.untracked.vstring("0x1FFFFFFF-0x1600586e")),
        cms.PSet(detSelection = cms.uint32(7), detLabel = cms.string("TIB + 1.6.2.5")  ,selection=cms.untracked.vstring("0x1FFFFFFF-0x160069e5")),
        cms.PSet(detSelection = cms.uint32(8), detLabel = cms.string("TIB + 1.6.2.5")  ,selection=cms.untracked.vstring("0x1FFFFFFF-0x160069e6")),
        cms.PSet(detSelection = cms.uint32(9), detLabel = cms.string("TIB + 1.6.2.5")  ,selection=cms.untracked.vstring("0x1FFFFFFF-0x160069e9")),
        cms.PSet(detSelection = cms.uint32(10),detLabel = cms.string("TIB + 1.6.2.5")  ,selection=cms.untracked.vstring("0x1FFFFFFF-0x160069ea")),
        cms.PSet(detSelection = cms.uint32(11),detLabel = cms.string("TIB + 1.6.2.5")  ,selection=cms.untracked.vstring("0x1FFFFFFF-0x160069ed")),
        cms.PSet(detSelection = cms.uint32(12),detLabel = cms.string("TIB + 1.6.2.5")  ,selection=cms.untracked.vstring("0x1FFFFFFF-0x160069ee")),
        cms.PSet(detSelection = cms.uint32(13),detLabel = cms.string("TOB + 1.3.1.6")  ,selection=cms.untracked.vstring("0x1FFFFFFF-0x1a0062c5")),
        cms.PSet(detSelection = cms.uint32(14),detLabel = cms.string("TOB + 1.3.1.6")  ,selection=cms.untracked.vstring("0x1FFFFFFF-0x1a0062c6")),
        cms.PSet(detSelection = cms.uint32(15),detLabel = cms.string("TOB + 1.3.1.6")  ,selection=cms.untracked.vstring("0x1FFFFFFF-0x1a0062c9")),
        cms.PSet(detSelection = cms.uint32(16),detLabel = cms.string("TOB + 1.3.1.6")  ,selection=cms.untracked.vstring("0x1FFFFFFF-0x1a0062ca")),
        cms.PSet(detSelection = cms.uint32(17),detLabel = cms.string("TOB + 1.3.1.6")  ,selection=cms.untracked.vstring("0x1FFFFFFF-0x1a0062cd")),
        cms.PSet(detSelection = cms.uint32(18),detLabel = cms.string("TOB + 1.3.1.6")  ,selection=cms.untracked.vstring("0x1FFFFFFF-0x1a0062ce")),
        cms.PSet(detSelection = cms.uint32(19),detLabel = cms.string("TOB + 1.3.1.6")  ,selection=cms.untracked.vstring("0x1FFFFFFF-0x1a0062d1")),
        cms.PSet(detSelection = cms.uint32(20),detLabel = cms.string("TOB + 1.3.1.6")  ,selection=cms.untracked.vstring("0x1FFFFFFF-0x1a0062d2")),
        cms.PSet(detSelection = cms.uint32(21),detLabel = cms.string("TOB + 1.3.1.6")  ,selection=cms.untracked.vstring("0x1FFFFFFF-0x1a0062d5")),
        cms.PSet(detSelection = cms.uint32(22),detLabel = cms.string("TOB + 1.3.1.6")  ,selection=cms.untracked.vstring("0x1FFFFFFF-0x1a0062d6")),
        cms.PSet(detSelection = cms.uint32(23),detLabel = cms.string("TOB + 1.3.1.6")  ,selection=cms.untracked.vstring("0x1FFFFFFF-0x1a0062d9")),
        cms.PSet(detSelection = cms.uint32(24),detLabel = cms.string("TOB + 1.3.1.6")  ,selection=cms.untracked.vstring("0x1FFFFFFF-0x1a0062da")),             
        cms.PSet(detSelection = cms.uint32(25),detLabel = cms.string("TOB + 4.3.3.8")  ,selection=cms.untracked.vstring("0x1FFFFFFF-0x1a0120a4")),
        cms.PSet(detSelection = cms.uint32(26),detLabel = cms.string("TOB + 4.3.3.8")  ,selection=cms.untracked.vstring("0x1FFFFFFF-0x1a0120a8")),
        cms.PSet(detSelection = cms.uint32(27),detLabel = cms.string("TOB + 4.3.3.8")  ,selection=cms.untracked.vstring("0x1FFFFFFF-0x1a0120ac")),
        cms.PSet(detSelection = cms.uint32(28),detLabel = cms.string("TOB + 4.3.3.8")  ,selection=cms.untracked.vstring("0x1FFFFFFF-0x1a0120b0")),
        cms.PSet(detSelection = cms.uint32(29),detLabel = cms.string("TOB + 4.3.3.8")  ,selection=cms.untracked.vstring("0x1FFFFFFF-0x1a0120b4")),
        cms.PSet(detSelection = cms.uint32(30),detLabel = cms.string("TOB + 4.3.3.8")  ,selection=cms.untracked.vstring("0x1FFFFFFF-0x1a0120b8")),
        cms.PSet(detSelection = cms.uint32(31),detLabel = cms.string("TEC - 3.7.1.1.2"),selection=cms.untracked.vstring("0x1FFFFFFF-0x1c05e464")),
        cms.PSet(detSelection = cms.uint32(32),detLabel = cms.string("TEC - 3.7.1.1.2"),selection=cms.untracked.vstring("0x1FFFFFFF-0x1c05e468")),
        cms.PSet(detSelection = cms.uint32(33),detLabel = cms.string("TEC - 3.7.1.1.2"),selection=cms.untracked.vstring("0x1FFFFFFF-0x1c05e46c")),
        cms.PSet(detSelection = cms.uint32(34),detLabel = cms.string("TEC - 3.7.1.1.2"),selection=cms.untracked.vstring("0x1FFFFFFF-0x1c05e484")),
        cms.PSet(detSelection = cms.uint32(35),detLabel = cms.string("TEC - 3.7.1.1.2"),selection=cms.untracked.vstring("0x1FFFFFFF-0x1c05e488")),
        cms.PSet(detSelection = cms.uint32(36),detLabel = cms.string("TEC - 3.7.1.1.2"),selection=cms.untracked.vstring("0x1FFFFFFF-0x1c05e48c")),
        cms.PSet(detSelection = cms.uint32(37),detLabel = cms.string("TEC - 3.7.1.1.2"),selection=cms.untracked.vstring("0x1FFFFFFF-0x1c05e490")),
        cms.PSet(detSelection = cms.uint32(38),detLabel = cms.string("TEC - 3.7.1.1.2"),selection=cms.untracked.vstring("0x1FFFFFFF-0x1c05e4c4")),
        cms.PSet(detSelection = cms.uint32(39),detLabel = cms.string("TEC - 3.7.1.1.2"),selection=cms.untracked.vstring("0x1FFFFFFF-0x1c05e4c8")),
        cms.PSet(detSelection = cms.uint32(40),detLabel = cms.string("TEC - 3.7.1.1.2"),selection=cms.untracked.vstring("0x1FFFFFFF-0x1c05e4cc")),
        cms.PSet(detSelection = cms.uint32(41),detLabel = cms.string("TEC - 3.7.1.1.2"),selection=cms.untracked.vstring("0x1FFFFFFF-0x1c05e4d0")),
        cms.PSet(detSelection = cms.uint32(42),detLabel = cms.string("TEC - 3.7.1.1.3"),selection=cms.untracked.vstring("0x1FFFFFFF-0x1c05e4a5")),
        cms.PSet(detSelection = cms.uint32(43),detLabel = cms.string("TEC - 3.7.1.1.3"),selection=cms.untracked.vstring("0x1FFFFFFF-0x1c05e4a6")),
        cms.PSet(detSelection = cms.uint32(44),detLabel = cms.string("TEC - 3.7.1.1.3"),selection=cms.untracked.vstring("0x1FFFFFFF-0x1c05e4a9")),
        cms.PSet(detSelection = cms.uint32(45),detLabel = cms.string("TEC - 3.7.1.1.3"),selection=cms.untracked.vstring("0x1FFFFFFF-0x1c05e4aa")),
        cms.PSet(detSelection = cms.uint32(46),detLabel = cms.string("TEC - 3.7.1.1.3"),selection=cms.untracked.vstring("0x1FFFFFFF-0x1c05e4e4")),
        cms.PSet(detSelection = cms.uint32(47),detLabel = cms.string("TEC - 3.7.1.1.3"),selection=cms.untracked.vstring("0x1FFFFFFF-0x1c05e4e8")),
        cms.PSet(detSelection = cms.uint32(48),detLabel = cms.string("TEC - 3.7.1.1.3"),selection=cms.untracked.vstring("0x1FFFFFFF-0x1c05e4ec")),
        cms.PSet(detSelection = cms.uint32(49),detLabel = cms.string("TEC - 3.7.1.1.3"),selection=cms.untracked.vstring("0x1FFFFFFF-0x1c05e4f0")),
        cms.PSet(detSelection = cms.uint32(50),detLabel = cms.string("TEC - 3.7.1.1.3"),selection=cms.untracked.vstring("0x1FFFFFFF-0x1c05e4f4"))
        )
                                                                                                                       )

################################################################################################
#TRACK PRODUCER
#now we give the TrackCandidate coming out of the CalibrationTrackSelectorFromDetIdList to the track producer
################################################################################################
import RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cff   
HitFilteredTracks = RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cff.ctfWithMaterialTracks.clone(
    src = 'ALCARECOSiStripCalSmallBiasScanSelectedTracks',
    #TrajectoryInEvent = True
    TTRHBuilder = "WithAngleAndTemplate"
    )

ALCARECOTrackFilterRefit = cms.Sequence(offlineBeamSpot +
                                        ALCARECOSiStripCalSmallBiasScanTracksRefit + 
                                        ALCARECOSiStripCalSmallBiasScanSelectedTracks +
                                        HitFilteredTracks 
                                        )
# Select only good tracks
import Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi
ALCARECOSiStripCalSmallBiasScan = Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi.AlignmentTrackSelector.clone()

ALCARECOSiStripCalSmallBiasScan.filter         = True ##do not store empty events	
ALCARECOSiStripCalSmallBiasScan.src            = 'HitFilteredTracks'
ALCARECOSiStripCalSmallBiasScan.applyBasicCuts = True
ALCARECOSiStripCalSmallBiasScan.ptMin          = 0.8 ##GeV
ALCARECOSiStripCalSmallBiasScan.nHitMin        = 6 ## at least 6 hits required
ALCARECOSiStripCalSmallBiasScan.chi2nMax       = 10.

ALCARECOSiStripCalSmallBiasScan.GlobalSelector.applyIsolationtest    = False
ALCARECOSiStripCalSmallBiasScan.GlobalSelector.applyGlobalMuonFilter = False
ALCARECOSiStripCalSmallBiasScan.GlobalSelector.applyJetCountFilter   = False

ALCARECOSiStripCalSmallBiasScan.TwoBodyDecaySelector.applyMassrangeFilter    = False
ALCARECOSiStripCalSmallBiasScan.TwoBodyDecaySelector.applyChargeFilter       = False
ALCARECOSiStripCalSmallBiasScan.TwoBodyDecaySelector.applyAcoplanarityFilter = False
ALCARECOSiStripCalSmallBiasScan.TwoBodyDecaySelector.applyMissingETFilter    = False

# Final Sequence #
seqALCARECOSiStripCalSmallBiasScan = cms.Sequence(ALCARECOSiStripCalSmallBiasScanHLT*DCSStatusForSiStripCalSmallBiasScan*ALCARECOTrackFilterRefit*ALCARECOSiStripCalSmallBiasScan)
