import FWCore.ParameterSet.Config as cms

from CalibMuon.CSCCalibration.CSCIndexer_cfi import (
    CSCIndexerESSource as _CSCIndexerESSource,
)

hltPhase2CSCIndexerESSource = _CSCIndexerESSource.clone()
