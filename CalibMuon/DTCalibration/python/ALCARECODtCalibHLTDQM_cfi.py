import FWCore.ParameterSet.Config as cms

selectionName = 'DtCalibSynch'
from DQM.HLTEvF.HLTMonBitSummary_cfi import hltMonBitSummary
from CalibMuon.DTCalibration.ALCARECODtCalib_cff import ALCARECODtCalibHLTFilter as ALCARECODtCalibHLTFilterRef
ALCARECODtCalibHLTDQM = hltMonBitSummary.clone(
    directory = "AlCaReco/" + selectionName + "/HLTSummary",
    histLabel = selectionName,
    HLTPaths = ["HLT_.*Mu.*"],
    eventSetupPathsKey =  ALCARECODtCalibHLTFilterRef.eventSetupPathsKey.value()
)
