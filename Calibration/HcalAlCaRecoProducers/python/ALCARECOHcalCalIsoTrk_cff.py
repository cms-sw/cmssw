import FWCore.ParameterSet.Config as cms

#------------------------------------------------
#AlCaReco filtering for HCAL isotrk:
#------------------------------------------------
import HLTrigger.HLTfilters.hltHighLevel_cfi
from Calibration.HcalAlCaRecoProducers.alcaisotrk_cfi import *

isoHLT = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone(
#    HLTPaths = ['HLT_IsoTrack'],
    eventSetupPathsKey='HcalCalIsoTrk',
    throw = False #dont throw except on unknown path name

)


seqALCARECOHcalCalIsoTrk = cms.Sequence(isoHLT*IsoProd)




