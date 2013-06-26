import FWCore.ParameterSet.Config as cms

from EventFilter.HcalRawToDigi.hcallasereventfilter2012_cfi import *

hcallLaserEvent2012Filter = cms.Sequence(hcallasereventfilter2012)
