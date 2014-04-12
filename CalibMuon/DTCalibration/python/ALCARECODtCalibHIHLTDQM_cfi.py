import FWCore.ParameterSet.Config as cms

selectionName = 'DtCalibSynchHI'
from DQM.HLTEvF.HLTMonBitSummary_cfi import hltMonBitSummary
from CalibMuon.DTCalibration.ALCARECODtCalibHI_cff import ALCARECODtCalibHIHLTFilter as ALCARECODtCalibHIHLTFilterRef
ALCARECODtCalibHIHLTDQM = hltMonBitSummary.clone(
    directory = "AlCaReco/" + selectionName + "/HLTSummary",
    histLabel = selectionName,
    HLTPaths = ["HLT_.*Mu.*"],
    eventSetupPathsKey =  ALCARECODtCalibHIHLTFilterRef.eventSetupPathsKey.value()
)
