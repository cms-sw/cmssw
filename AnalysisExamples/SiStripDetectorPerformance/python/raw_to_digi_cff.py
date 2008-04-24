import FWCore.ParameterSet.Config as cms

# SiStripRaw2digi
from EventFilter.SiStripRawToDigi.SiStripDigis_cfi import *
#include "EventFilter/SiStripRawToDigi/data/FedChannelDigis.cfi"
# ZeroSuppressor
from RecoLocalTracker.SiStripZeroSuppression.SiStripZeroSuppression_cfi import *
# produce SiStripFecCabling and SiStripDetCabling out of SiStripFedCabling
sistripconn = cms.ESProducer("SiStripConnectivity")

raw_to_digi = cms.Sequence(siStripDigis*siStripZeroSuppression)

