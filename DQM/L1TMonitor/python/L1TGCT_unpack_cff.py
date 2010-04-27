import FWCore.ParameterSet.Config as cms

# Take defaults except unpack 5 BXs for monitoring - A. Tapper 2 April 2010
from EventFilter.GctRawToDigi.l1GctHwDigis_cfi import *
l1GctHwDigis.numberOfGctSamplesToUnpack = cms.uint32(5)

