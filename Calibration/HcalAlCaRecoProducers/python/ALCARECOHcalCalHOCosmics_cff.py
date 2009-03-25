import FWCore.ParameterSet.Config as cms

#------------------------------------------------
#AlCaReco filtering for HCAL HO:
#------------------------------------------------
from Calibration.HcalAlCaRecoProducers.alcahomuoncosmics_cfi import *
import HLTrigger.HLTfilters.hltHighLevel_cfi


ALCARECOHcalCalHOCosmicHLT = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone(
#   HLTPaths = ['HLT_L1MuOpen', 'HLT_Mu3',  'HLT_Mu5'],
   eventSetupPathsKey='HcalCalHOCosmics', 
   throw = False #dont throw except on unknown path name
) 

seqALCARECOHcalCalHOCosmics = cms.Sequence(ALCARECOHcalCalHOCosmicHLT*hoCalibCosmicsProducer)


