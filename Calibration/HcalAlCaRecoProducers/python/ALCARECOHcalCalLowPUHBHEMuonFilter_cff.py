import FWCore.ParameterSet.Config as cms

#-------------------------------------------------
#AlCaReco filtering for HCAL LowPU HBHEMuon:
#-------------------------------------------------
import HLTrigger.HLTfilters.hltHighLevel_cfi
ALCARECOHcalCalLowPUHBHEMuonFilterHLT = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone(
    eventSetupPathsKey = 'HcalCalHBHEMuonFilter',
    throw = False #dont throw except on unknown path name
)

from Calibration.HcalAlCaRecoProducers.alcaLowPUHBHEMuonFilter_cfi import *

seqALCARECOHcalCalLowPUHBHEMuonFilter = cms.Sequence(ALCARECOHcalCalLowPUHBHEMuonFilterHLT *
                                                     alcaLowPUHBHEMuonFilter)
