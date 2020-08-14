import FWCore.ParameterSet.Config as cms

from CalibMuon.CSCCalibration.CSCChannelMapper_cfi import (
    CSCChannelMapperESSource as _CSCChannelMapperESSource,
)

hltPhase2CSCChannelMapperESSource = _CSCChannelMapperESSource.clone()
