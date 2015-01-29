import FWCore.ParameterSet.Config as cms

from RecoMuon.MuonIdentification.DTTimingExtractor_cfi import *
from RecoMuon.MuonIdentification.CSCTimingExtractor_cfi import *

TimingFillerBlock = cms.PSet(
  TimingFillerParameters = cms.PSet(
    DTTimingExtractorBlock,
    CSCTimingExtractorBlock,
    
    # Ecal minimum energy cut
    EcalEnergyCut = cms.double(0.4),
    # Ecal error parametrization
    ErrorEB = cms.double(2.085),
    ErrorEE = cms.double(6.95),
    
    # On/off switches for combined time measurement
    UseDT  = cms.bool(True),
    UseCSC = cms.bool(True),
    UseECAL= cms.bool(False)
  )
)


