import FWCore.ParameterSet.Config as cms

from EventFilter.HcalRawToDigi.HcalLaserEventFilterProducer_cfi import *

hcallLaserEventFilterResultSeq = cms.Sequence(HcalLaserEventFilterResult)
