import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.Reconstruction_cff import *

particleFlowRecHitHCAL.LongShortFibre_Cut = 30.
particleFlowRecHitHCAL.ApplyPulseDPG = True

hcalRecAlgos.SeverityLevels[3].RecHitFlags.remove("HFDigiTime")
hcalRecAlgos.SeverityLevels[4].RecHitFlags.append("HFDigiTime")

CSCHaloData.ExpectedBX = cms.int32(3)

ecalGlobalUncalibRecHit.doEBtimeCorrection = True
ecalGlobalUncalibRecHit.doEEtimeCorrection = True
