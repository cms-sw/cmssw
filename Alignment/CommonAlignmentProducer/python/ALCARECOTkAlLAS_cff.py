# AlCaReco for laser alignment system
import FWCore.ParameterSet.Config as cms

# Need to add this module to get the 'scalersRawToDigi' product
# for the DCS bit filter
from EventFilter.ScalersRawToDigi.ScalersRawToDigi_cfi import scalersRawToDigi

# DCS partitions
# "EBp","EBm","EEp","EEm","HBHEa","HBHEb","HBHEc","HF","HO","RPC"
# "DT0","DTp","DTm","CSCp","CSCm","CASTOR","TIBTID","TOB","TECp","TECm"
# "BPIX","FPIX","ESp","ESm"
import DPGAnalysis.Skims.skim_detstatus_cfi
ALCARECOTkAlLASDCSFilter = DPGAnalysis.Skims.skim_detstatus_cfi.dcsstatus.clone(
    DetectorType = cms.vstring('TIBTID','TOB','TECp','TECm'),
    ApplyFilter  = cms.bool(True),
    AndOr        = cms.bool(True),
    DebugOn      = cms.untracked.bool(False)
)

import EventFilter.SiStripRawToDigi.SiStripDigis_cfi
ALCARECOTkAlLASsiStripDigis = EventFilter.SiStripRawToDigi.SiStripDigis_cfi.siStripDigis.clone(
  ProductLabel = 'hltTrackerCalibrationRaw'
#  ProductLabel = 'source'
)

import Alignment.LaserAlignment.LaserAlignmentEventFilter_cfi
ALCARECOTkAlLASEventFilter = Alignment.LaserAlignment.LaserAlignmentEventFilter_cfi.LaserAlignmentEventFilter.clone()

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

seqALCARECOTkAlLAS = cms.Sequence(scalersRawToDigi+ALCARECOTkAlLASDCSFilter+ALCARECOTkAlLASsiStripDigis+ALCARECOTkAlLASEventFilter+ALCARECOTkAlLAST0Producer)

