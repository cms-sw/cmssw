import FWCore.ParameterSet.Config as cms

selectionName = 'DtCalibSynchCosmics'
from DQM.HLTEvF.HLTMonBitSummary_cfi import hltMonBitSummary
from CalibMuon.DTCalibration.ALCARECODtCalibCosmics_cff import ALCARECODtCalibCosmicsHLTFilter as ALCARECODtCalibCosmicsHLTFilterRef
ALCARECODtCalibCosmicsHLTDQM = hltMonBitSummary.clone(
    directory = "AlCaReco/" + selectionName + "/HLTSummary",
    histLabel = selectionName,
    HLTPaths = ["HLT_.*Mu.*"],
    eventSetupPathsKey =  ALCARECODtCalibCosmicsHLTFilterRef.eventSetupPathsKey.value()
)
