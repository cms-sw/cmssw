import FWCore.ParameterSet.Config as cms

from DQM.DTMonitorModule.dtDataIntegrityTask_cfi import *

import EventFilter.DTRawToDigi.dturosunpacker_cfi
dtunpacker = EventFilter.DTRawToDigi.dturosunpacker_cfi.dturosunpacker.clone()


