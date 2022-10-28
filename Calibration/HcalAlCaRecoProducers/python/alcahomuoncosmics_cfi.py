import FWCore.ParameterSet.Config as cms

#-----------------------------------------------------------
#AlCaReco Filtering for HO calibration using cosmicMuon/StandAlonMuon
#----------------------------------------------------------- 
#process.load('Configuration.StandardSequences.Reconstruction_Data_cff')
#from Configuration.StandardSequences.Reconstruction_Data_cff import *
from RecoLocalCalo.HcalRecAlgos.hcalRecAlgoESProd_cfi import *

import Calibration.HcalAlCaRecoProducers.alcaHOCalibProducer_cfi
hoCalibCosmicsProducer = Calibration.HcalAlCaRecoProducers.alcaHOCalibProducer_cfi.alcaHOCalibProducer.clone(
    CosmicData = True,
    muons = cms.untracked.InputTag("cosmicMuons")
)

