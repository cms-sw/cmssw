import FWCore.ParameterSet.Config as cms

from RecoMuon.MuonIdentification.DTTimingExtractor_cfi import *
from RecoMuon.MuonIdentification.CSCTimingExtractor_cfi import *

TimingFillerBlock = cms.PSet(
  TimingFillerParameters = cms.PSet(
    DTTimingExtractorBlock,
    CSCTimingExtractorBlock,
    
    # Single hit time measurement precition in ns
    ErrorDT  = cms.double(3.1),
    ErrorCSC = cms.double(7.),
    
    # Ecal minimum energy cut
    EcalEnergyCut = cms.double(0.4)
  )
)


