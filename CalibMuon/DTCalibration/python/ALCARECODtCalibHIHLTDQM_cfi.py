import FWCore.ParameterSet.Config as cms

selectionName = 'DtCalibSynchHI'
#from DQM.HLTEvF.HLTMonBitSummary_cfi import hltMonBitSummary
from CalibMuon.DTCalibration.ALCARECODtCalib_cff import ALCARECODtCalibHLTFilter
from CalibMuon.DTCalibration.ALCARECODtCalibHI_cff import ALCARECODtCalibHIHLTFilter as ALCARECODtCalibHIHLTFilterRef

