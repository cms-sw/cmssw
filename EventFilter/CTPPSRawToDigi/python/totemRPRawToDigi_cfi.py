import FWCore.ParameterSet.Config as cms

from EventFilter.CTPPSRawToDigi.totemVFATRawToDigi_cfi import totemVFATRawToDigi

totemRPRawToDigi = totemVFATRawToDigi.copy()
totemRPRawToDigi.subSystem = "TrackingStrip"
