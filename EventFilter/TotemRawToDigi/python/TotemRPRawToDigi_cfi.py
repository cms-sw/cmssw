import FWCore.ParameterSet.Config as cms

from EventFilter.TotemRawToDigi.TotemVFATRawToDigi_cfi import TotemVFATRawToDigi

TotemRPRawToDigi = TotemVFATRawToDigi.copy()
TotemRPRawToDigi.subSystem = "RP"
