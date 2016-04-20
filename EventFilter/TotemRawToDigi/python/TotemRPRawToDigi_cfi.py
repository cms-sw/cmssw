import FWCore.ParameterSet.Config as cms

from EventFilter.TotemRawToDigi.TotemVFATRawToDigi_cfi import totemVFATRawToDigi

totemRPRawToDigi = totemVFATRawToDigi.copy()
totemRPRawToDigi.subSystem = "RP"
