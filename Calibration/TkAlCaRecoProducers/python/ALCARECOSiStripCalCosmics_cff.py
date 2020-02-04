import FWCore.ParameterSet.Config as cms

# Set the HLT paths
import HLTrigger.HLTfilters.hltHighLevel_cfi
ALCARECOSiStripCalCosmicsHLT = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone(
    andOr = True, ## choose logical OR between Triggerbits
    eventSetupPathsKey = 'SiStripCalCosmics',
    throw = False # tolerate triggers stated above, but not available
    )

# Select only events where tracker had HV on (according to DCS bit information)
# AND respective partition is in the run (according to FED information)
import CalibTracker.SiStripCommon.SiStripDCSFilter_cfi
DCSStatusForSiStripCalCosmics = CalibTracker.SiStripCommon.SiStripDCSFilter_cfi.siStripDCSFilter.clone(
    DetectorType = cms.vstring('TIBTID','TOB','TECp','TECm'),
    ApplyFilter  = cms.bool(True),
    AndOr        = cms.bool(True),
    DebugOn      = cms.untracked.bool(False)
    )

# Select only good tracks
import Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi
ALCARECOSiStripCalCosmics = Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi.AlignmentTrackSelector.clone()

ALCARECOSiStripCalCosmics.filter         = True ##do not store empty events	
ALCARECOSiStripCalCosmics.src            = 'ctfWithMaterialTracksP5'
ALCARECOSiStripCalCosmics.applyBasicCuts = True
ALCARECOSiStripCalCosmics.ptMin          = 0. ##GeV
ALCARECOSiStripCalCosmics.nHitMin        = 6  ## at least 6 hits required
ALCARECOSiStripCalCosmics.chi2nMax       = 10.

ALCARECOSiStripCalCosmics.GlobalSelector.applyIsolationtest    = False
ALCARECOSiStripCalCosmics.GlobalSelector.applyGlobalMuonFilter = False
ALCARECOSiStripCalCosmics.GlobalSelector.applyJetCountFilter   = False

ALCARECOSiStripCalCosmics.TwoBodyDecaySelector.applyMassrangeFilter    = False
ALCARECOSiStripCalCosmics.TwoBodyDecaySelector.applyChargeFilter       = False
ALCARECOSiStripCalCosmics.TwoBodyDecaySelector.applyAcoplanarityFilter = False
ALCARECOSiStripCalCosmics.TwoBodyDecaySelector.applyMissingETFilter    = False

# Sequence #
seqALCARECOSiStripCalCosmics = cms.Sequence(ALCARECOSiStripCalCosmicsHLT*DCSStatusForSiStripCalCosmics*ALCARECOSiStripCalCosmics)
