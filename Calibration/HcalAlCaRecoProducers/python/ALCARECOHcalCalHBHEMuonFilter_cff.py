import FWCore.ParameterSet.Config as cms

#-------------------------------------------------
#AlCaReco filtering for HCAL HBHEMuon:
#-------------------------------------------------
import HLTrigger.HLTfilters.hltHighLevel_cfi
ALCARECOHcalCalHBHEMuonFilterHLT = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone(
    eventSetupPathsKey = 'HcalCalHBHEMuonFilter',
    throw = False #dont throw except on unknown path name
)

from Calibration.HcalAlCaRecoProducers.AlcaHBHEMuonFilter_cfi import *

seqALCARECOHcalCalHBHEMuonFilter = cms.Sequence(ALCARECOHcalCalHBHEMuonFilterHLT *
                                                AlcaHBHEMuonFilter)
