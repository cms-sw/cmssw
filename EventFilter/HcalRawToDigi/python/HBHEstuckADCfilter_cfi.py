import FWCore.ParameterSet.Config as cms

import EventFilter.HcalRawToDigi.hbhestuckADCfilter_cfi
stuckADCfilter =  EventFilter.HcalRawToDigi.hbhestuckADCfilter_cfi.hbhestuckADCfilter.clone()
stuckADCfilter.thresholdADC = 100
