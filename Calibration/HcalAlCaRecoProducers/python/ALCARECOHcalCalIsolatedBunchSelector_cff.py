import FWCore.ParameterSet.Config as cms

#-------------------------------------------------
#AlCaReco filtering for HCAL isotrk:isolated bunch:
#-------------------------------------------------
import HLTrigger.HLTfilters.hltHighLevel_cfi
ALCARECOHcalCalIsolatedBunchSelectorHLT = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone(
    eventSetupPathsKey = 'HcalCalIsolatedBunchSelector',
    throw = False #dont throw except on unknown path name
)

from Calibration.HcalAlCaRecoProducers.alcaIsolatedBunchSelector_cfi import *

seqALCARECOHcalCalIsolatedBunchSelector = cms.Sequence(ALCARECOHcalCalIsolatedBunchSelectorHLT *
                                                       AlcaIsolatedBunchSelector)
