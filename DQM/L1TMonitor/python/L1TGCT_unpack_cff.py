import FWCore.ParameterSet.Config as cms

# relevant FED is 745
# if the default source gets updated then we can get rid of this
# cff file and just point at the original one
from EventFilter.GctRawToDigi.l1GctHwDigis_cfi import *
l1GctHwDigis.inputLabel = 'source'

