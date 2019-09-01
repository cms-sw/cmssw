import FWCore.ParameterSet.Config as cms

#-------------------------------------------------
#AlCaReco filtering for HCAL isotrk:isolated bunch:
#-------------------------------------------------
import HLTrigger.HLTfilters.hltHighLevel_cfi
ALCARECOHcalCalIsolatedBunchFilterHLT = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone(
    eventSetupPathsKey = 'HcalCalIsolatedBunchFilter',
    throw = False #dont throw except on unknown path name
)

from Calibration.HcalAlCaRecoProducers.alcaIsolatedBunchFilter_cfi import *

seqALCARECOHcalCalIsolatedBunchFilter = cms.Sequence(ALCARECOHcalCalIsolatedBunchFilterHLT *
                                                     alcaIsolatedBunchFilter)
