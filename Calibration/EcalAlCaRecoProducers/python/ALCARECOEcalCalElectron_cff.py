import FWCore.ParameterSet.Config as cms

#
#  run on collection of electrons to make a collection of AlCaReco electrons 
#
from Calibration.EcalAlCaRecoProducers.alCaIsolatedElectrons_cfi import *
from Calibration.EcalAlCaRecoProducers.ewkHLTFilter_cfi import *
from Calibration.EcalAlCaRecoProducers.electronFilter_cfi import *
seqALCARECOEcalCalElectronRECO = cms.Sequence(alCaIsolatedElectrons)
seqALCARECOEcalCalElectron = cms.Sequence(ewkHLTFilter*electronFilter*seqALCARECOEcalCalElectronRECO) ##HLT selection: on

ewkHLTFilter.HLTPaths = ['HLT1Electron', 'HLT2Electron', 'HLT1ElectronRelaxed', 'HLT2ElectronRelaxed']

