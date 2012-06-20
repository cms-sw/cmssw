# AlCaReco for laser alignment system
import FWCore.ParameterSet.Config as cms

import EventFilter.SiStripRawToDigi.SiStripDigis_cfi
ALCARECOTkAlLASsiStripDigis = EventFilter.SiStripRawToDigi.SiStripDigis_cfi.siStripDigis.clone(
  ProductLabel = 'hltTrackerCalibrationRaw'
)

import Alignment.LaserAlignment.LaserAlignmentEventFilter_cfi
ALCARECOTkAlLASEventFilter = Alignment.LaserAlignment.LaserAlignmentEventFilter_cfi.LaserAlignmentEventFilter.clone(
  FedInputTag = 'hltTrackerCalibrationRaw'
)

import Alignment.LaserAlignment.LaserAlignmentT0Producer_cfi
ALCARECOTkAlLAST0Producer = Alignment.LaserAlignment.LaserAlignmentT0Producer_cfi.laserAlignmentT0Producer.clone(
  DigiProducerList = cms.VPSet(
    cms.PSet(
       DigiLabel = cms.string( 'ZeroSuppressed' ),
       DigiType = cms.string( 'Processed' ),
       DigiProducer = cms.string( 'ALCARECOTkAlLASsiStripDigis' )
    )
  )
)

seqALCARECOTkAlLAS = cms.Sequence(ALCARECOTkAlLASsiStripDigis+ALCARECOTkAlLASEventFilter+ALCARECOTkAlLAST0Producer)
