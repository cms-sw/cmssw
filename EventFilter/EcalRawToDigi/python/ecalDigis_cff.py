import FWCore.ParameterSet.Config as cms

# legacy raw to digi on the CPU
from EventFilter.EcalRawToDigi.EcalUnpackerData_cfi import ecalEBunpacker as _ecalEBunpacker
ecalDigis = _ecalEBunpacker.clone()

ecalDigisTask = cms.Task(ecalDigis)
