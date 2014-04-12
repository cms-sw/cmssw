import FWCore.ParameterSet.Config as cms

# ALCARECOZeeMCEcalCalElectron_Output.cff ###################
from Calibration.EcalAlCaRecoProducers.ALCARECOEcalCalElectron_Output_cff import *
from Calibration.EcalAlCaRecoProducers.zeeMCInfo_EventContent_cff import *
OutALCARECOEcalCalElectron.SelectEvents.SelectEvents = ['zeeHLTPath']
OutALCARECOEcalCalElectron.outputCommands.extend(MCInfo.outputCommands)

