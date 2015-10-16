import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.Reconstruction_cff import *



#HF cleaning for data in the new design
# adding q tests for those lines            
#particleFlowRecHitHCAL.LongShortFibre_Cut = 30.
#particleFlowRecHitHCAL.ApplyPulseDPG = True


for qTest in particleFlowRecHitHF.producers[0].qualityTests:
    if qTest.name == 'PFRecHitQTestHCALChannel':
        qTest.maxSeverities.append(9)
        qTest.cleaningThresholds.append(30.)
        qTest.flags.append('HFDigi')
             

import RecoLocalCalo.HcalRecAlgos.RemoveAddSevLevel as HcalRemoveAddSevLevel
HcalRemoveAddSevLevel.AddFlag(hcalRecAlgos,"HFDigiTime",11)
HcalRemoveAddSevLevel.AddFlag(hcalRecAlgos,"HBHEFlatNoise",12)
HcalRemoveAddSevLevel.AddFlag(hcalRecAlgos,"HBHENegativeNoise",12)

CSCHaloData.ExpectedBX = cms.int32(3)

from JetMETCorrections.Configuration.JetCorrectors_cff import ak4PFCHSResidualCorrector, ak4PFCHSL1FastL2L3ResidualCorrector, ak4PFCHSL1FastL2L3ResidualCorrectorChain
jetCorrectorsForReco.replace(ak4PFCHSL1FastL2L3CorrectorChain, ak4PFCHSL1FastL2L3ResidualCorrectorChain)
