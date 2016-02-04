import FWCore.ParameterSet.Config as cms


#
#  Old ALCA stream for electron calibration, used in CSA07
#  
#
#  run on collection of electrons to make a collection of AlCaReco electrons 
#
from Calibration.EcalAlCaRecoProducers.alCaIsolatedElectrons_cfi import *
from HLTrigger.HLTfilters.hltHighLevel_cfi import *
from Calibration.EcalAlCaRecoProducers.electronFilter_cfi import *
from RecoLocalCalo.EcalRecAlgos.EcalSeverityLevelESProducer_cfi import *

electronewkHLTFilter = hltHighLevel.clone()
electronewkHLTFilter.throw = False 
electronewkHLTFilter.HLTPaths = ['HLT_Ele15_SW_L1R', 'HLT_DoubleEle10_SW_L1R', # 1E31
                                 'HLT_Ele10_LW_L1R', 'HLT_DoubleEle5_SW_L1R']  # 8E29

seqALCARECOEcalCalElectronRECO = cms.Sequence(alCaIsolatedElectrons)
seqALCARECOEcalCalElectron = cms.Sequence(electronewkHLTFilter*electronFilter*seqALCARECOEcalCalElectronRECO) 
