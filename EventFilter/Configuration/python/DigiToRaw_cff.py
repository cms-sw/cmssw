import FWCore.ParameterSet.Config as cms

#
# David Lange, LLNL 
# February 26, 2007
#
# Definition of DigiToRaw sequence
#
# Pixel DigiToRaw conversion
#  include "EventFilter/SiPixelRawToDigi/data/SiPixelDigiToRaw.cfi"
# Strip DigiToRaw conversion
#  include "EventFilter/SiStriPRawToDigi/data/DigiToRaw.cfi"
# ECAL
# HCAL
# DT conversion
# RPC
# CSC
# CSCTF
# Gct
# etc, etc
#
# Collect all this together
rawDataCollector = cms.EDFilter("RawDataCollectorModule")

DigiToRaw = cms.Sequence(rawDataCollector)

