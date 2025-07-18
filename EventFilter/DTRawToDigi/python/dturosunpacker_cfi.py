import FWCore.ParameterSet.Config as cms

from EventFilter.DTRawToDigi.dTuROSRawToDigi_cfi import dTuROSRawToDigi as _dTuROSRawToDigi
dturosunpacker = _dTuROSRawToDigi.clone()
