import FWCore.ParameterSet.Config as cms

# 1. HLT filter
#------------------------------------------------------
import HLTrigger.HLTfilters.hltHighLevel_cfi as _hlt
ALCARECOPPSCalMaxTracksFilter = _hlt.hltHighLevel.clone(eventSetupPathsKey = 'PPSCalMaxTracks')


# 2. RAW2DIGI
#------------------------------------------------------

from EventFilter.CTPPSRawToDigi.ctppsRawToDigi_cff import *
from EventFilter.CTPPSRawToDigi.ctppsRawToDigi_cff import ctppsDiamondRawToDigi as _ctppsDiamondRawToDigi
from EventFilter.CTPPSRawToDigi.ctppsRawToDigi_cff import totemTimingRawToDigi as _totemTimingRawToDigi
from EventFilter.CTPPSRawToDigi.ctppsRawToDigi_cff import ctppsPixelDigis as _ctppsPixelDigis

ctppsDiamondRawToDigiAlCaRecoProducer  = _ctppsDiamondRawToDigi.clone(rawDataTag = 'hltPPSCalibrationRaw')
totemTimingRawToDigiAlCaRecoProducer = _totemTimingRawToDigi.clone(rawDataTag = 'hltPPSCalibrationRaw')
ctppsPixelDigisAlCaRecoProducer = _ctppsPixelDigis.clone(inputLabel = 'hltPPSCalibrationRaw')

ctppsRawToDigiTaskAlCaRecoProducer = cms.Task(
    ctppsDiamondRawToDigiAlCaRecoProducer,
    totemTimingRawToDigiAlCaRecoProducer,
    ctppsPixelDigisAlCaRecoProducer
)

ALCARECOPPSCalMaxTracksRaw2Digi = cms.Sequence(ctppsRawToDigiTaskAlCaRecoProducer)

# 3 a). RECO - PIXELS
#------------------------------------------------------

from RecoPPS.Configuration.recoCTPPS_cff import *
from RecoPPS.Local.ctppsPixelClusters_cfi import ctppsPixelClusters as _ctppsPixelClusters
from RecoPPS.Local.ctppsPixelRecHits_cfi import ctppsPixelRecHits as _ctppsPixelRecHits
from RecoPPS.Local.ctppsPixelLocalTracks_cfi import ctppsPixelLocalTracks as _ctppsPixelLocalTracks

ctppsPixelClustersAlCaRecoProducer = _ctppsPixelClusters.clone(tag = 'ctppsPixelDigisAlCaRecoProducer')
ctppsPixelRecHitsAlCaRecoProducer = _ctppsPixelRecHits.clone(RPixClusterTag = 'ctppsPixelClustersAlCaRecoProducer')
ctppsPixelLocalTracksAlCaRecoProducer = _ctppsPixelLocalTracks.clone(tag = 'ctppsPixelRecHitsAlCaRecoProducer')

ctppsPixelLocalReconstructionTaskAlCaRecoProducer = cms.Task(
    ctppsPixelClustersAlCaRecoProducer,
    ctppsPixelRecHitsAlCaRecoProducer,
    ctppsPixelLocalTracksAlCaRecoProducer
)

# 3 b). RECO - CTPPS DIAMONDS
#------------------------------------------------------

from RecoPPS.Local.ctppsDiamondRecHits_cfi import ctppsDiamondRecHits as _ctppsDiamondRecHits
from RecoPPS.Local.ctppsDiamondLocalTracks_cfi import ctppsDiamondLocalTracks as _ctppsDiamondLocalTracks
ctppsDiamondRecHitsAlCaRecoProducer = _ctppsDiamondRecHits.clone(digiTag = 'ctppsDiamondRawToDigiAlCaRecoProducer:TimingDiamond')
ctppsDiamondLocalTracksAlCaRecoProducer = _ctppsDiamondLocalTracks.clone(recHitsTag = 'ctppsDiamondRecHitsAlCaRecoProducer')

ctppsDiamondLocalReconstructionTaskAlCaRecoProducer = cms.Task(
    ctppsDiamondRecHitsAlCaRecoProducer,
    ctppsDiamondLocalTracksAlCaRecoProducer
)

# 3 c). RECO - TIMING DIAMONDS
#------------------------------------------------------

from RecoPPS.Local.totemTimingRecHits_cfi import totemTimingRecHits as _totemTimingRecHits
from RecoPPS.Local.diamondSampicLocalTracks_cfi import diamondSampicLocalTracks as _diamondSampicLocalTracks

totemTimingRecHitsAlCaRecoProducer = _totemTimingRecHits.clone(digiTag = 'totemTimingRawToDigiAlCaRecoProducer:TotemTiming')
diamondSampicLocalTracksAlCaRecoProducer = _diamondSampicLocalTracks.clone(recHitsTag = 'totemTimingRecHitsAlCaRecoProducer')

diamondSampicLocalReconstructionTaskAlCaRecoProducer = cms.Task(
    totemTimingRecHitsAlCaRecoProducer,
    diamondSampicLocalTracksAlCaRecoProducer
)

# 4. RECO - TRACKS and PROTONS
#------------------------------------------------------

from RecoPPS.Local.ctppsLocalTrackLiteProducer_cff import ctppsLocalTrackLiteProducer as _ctppsLocalTrackLiteProducer

ctppsLocalTrackLiteProducerAlCaRecoProducer = _ctppsLocalTrackLiteProducer.clone(
    includeStrips = False,
    includeDiamonds = True,
    includePixels = True,
    tagDiamondTrack = 'ctppsDiamondLocalTracksAlCaRecoProducer',
    tagPixelTrack = 'ctppsPixelLocalTracksAlCaRecoProducer'
)

from RecoPPS.ProtonReconstruction.ctppsProtons_cff import ctppsProtons as _ctppsProtons
ctppsProtonsAlCaRecoProducer = _ctppsProtons.clone(tagLocalTrackLite = 'ctppsLocalTrackLiteProducerAlCaRecoProducer')

# 5. RECO - final task assembly
#------------------------------------------------------

recoPPSTaskAlCaRecoProducer = cms.Task(
    ctppsDiamondLocalReconstructionTaskAlCaRecoProducer ,
    diamondSampicLocalReconstructionTaskAlCaRecoProducer ,
    ctppsPixelLocalReconstructionTaskAlCaRecoProducer ,
    ctppsLocalTrackLiteProducerAlCaRecoProducer ,
    ctppsProtonsAlCaRecoProducer
)
recoPPSSequenceAlCaRecoProducer = cms.Sequence(recoPPSTaskAlCaRecoProducer)

# 6. master sequence object
#------------------------------------------------------

seqALCARECOPPSCalMaxTracksReco = cms.Sequence( ALCARECOPPSCalMaxTracksFilter  + ALCARECOPPSCalMaxTracksRaw2Digi + recoPPSSequenceAlCaRecoProducer)