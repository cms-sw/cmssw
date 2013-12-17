import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.Reconstruction_cff import *

particleFlowRecHitHCAL.LongShortFibre_Cut = 30.
particleFlowRecHitHCAL.ApplyPulseDPG = True

import RecoLocalCalo.HcalRecAlgos.RemoveAddSevLevel as HcalRemoveAddSevLevel
HcalRemoveAddSevLevel.AddFlag(hcalRecAlgos,"HFDigiTime",11)
HcalRemoveAddSevLevel.AddFlag(hcalRecAlgos,"HBHEFlatNoise",12)
HcalRemoveAddSevLevel.AddFlag(hcalRecAlgos,"HBHESpikeNoise",12)

CSCHaloData.ExpectedBX = cms.int32(3)
