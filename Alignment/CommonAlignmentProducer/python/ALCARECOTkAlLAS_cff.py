import FWCore.ParameterSet.Config as cms

#---------------------------------------------------
# AlCaReco filtering for the Tracker Laser ALignment
#---------------------------------------------------

# need to run the digitizer on raw strip data first
import EventFilter.SiStripRawToDigi.SiStripDigis_cfi
TkAlLASsiStripDigis = EventFilter.SiStripRawToDigi.SiStripDigis_cfi.siStripDigis.clone(
  ProductLabel = cms.untracked.string( 'source' )
)

# redefine the input digis according to the clone's name
import Alignment.LaserAlignment.LaserAlignmentT0Producer_cfi
laserAlignmentT0Producer = Alignment.LaserAlignment.LaserAlignmentT0Producer_cfi.laserAlignmentT0Producer.clone(
  DigiProducerList = cms.VPSet(
    cms.PSet(
      DigiLabel = cms.string( 'ZeroSuppressed' ),
      DigiType = cms.string( 'Processed' ),
      DigiProducer = cms.string( 'TkAlLASsiStripDigis' )
    )
  )
)

seqALCARECOTkAlLAS = cms.Sequence( TkAlLASsiStripDigis + laserAlignmentT0Producer )

