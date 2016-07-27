import FWCore.ParameterSet.Config as cms

from EventFilter.TotemRawToDigi.totemVFATRawToDigi_cfi import totemVFATRawToDigi

totemRPRawToDigi = totemVFATRawToDigi.copy()
totemRPRawToDigi.subSystem = "RP"
