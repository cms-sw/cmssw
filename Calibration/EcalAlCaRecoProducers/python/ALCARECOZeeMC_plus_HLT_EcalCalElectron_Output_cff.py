import FWCore.ParameterSet.Config as cms

# ALCARECOZeeMCEcalCalElectron_Output.cff ###################
from Calibration.EcalAlCaRecoProducers.ALCARECOEcalCalElectron_Output_cff import *
from Calibration.EcalAlCaRecoProducers.zeeMCInfo_EventContent_cff import *
from Calibration.EcalAlCaRecoProducers.zeeHLTInfo_EventContent_cff import *
OutALCARECOEcalCalElectron.outputCommands.extend(MCInfo.outputCommands)
OutALCARECOEcalCalElectron.outputCommands.extend(HLTInfo.outputCommands)

