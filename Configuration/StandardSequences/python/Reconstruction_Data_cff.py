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
             
#--- Initial (Run1) HCAL data-specific flags customization
import RecoLocalCalo.HcalRecAlgos.RemoveAddSevLevel as HcalRemoveAddSevLevel
HcalRemoveAddSevLevel.AddFlag(hcalRecAlgos,"HFDigiTime",11,verbose=False)
HcalRemoveAddSevLevel.AddFlag(hcalRecAlgos,"HBHEFlatNoise",12)
HcalRemoveAddSevLevel.AddFlag(hcalRecAlgos,"HBHENegativeNoise",12)

#--- Subsequent era-wise HCAL data-specific flags customization 

from Configuration.Eras.Modifier_run2_25ns_specific_cff import run2_25ns_specific
def _modName(algos):
   HcalRemoveAddSevLevel.AddFlag(algos,"HBHEFlatNoise",8)
   HcalRemoveAddSevLevel.AddFlag(algos,"HFDigiTime",8)
run2_25ns_specific.toModify(hcalRecAlgos, _modName)

from Configuration.Eras.Modifier_run2_HCAL_2017_cff import run2_HCAL_2017
def _modName(algos):
   HcalRemoveAddSevLevel.RemoveFlag(algos,"HFDigiTime")  
run2_HCAL_2017.toModify(hcalRecAlgos, _modName)

#--- NB: MC and data get back in sync for >= Run3  ------------------------
from Configuration.Eras.Modifier_run3_HB_cff import run3_HB
def _modName(algos):
   HcalRemoveAddSevLevel.AddFlag(algos,"HBHENegativeNoise",8)
run3_HB.toModify(hcalRecAlgos, _modName)


CSCHaloData.ExpectedBX = cms.int32(3)

from JetMETCorrections.Configuration.JetCorrectors_cff import ak4PFCHSL1FastL2L3ResidualCorrectorTask, ak4PFCHSL1FastL2L3ResidualCorrectorTask

from JetMETCorrections.Configuration.JetCorrectors_cff import ak4PFCHSResidualCorrector, ak4PFCHSL1FastL2L3ResidualCorrector
jetCorrectorsForRecoTask.replace(ak4PFCHSL1FastL2L3CorrectorTask, ak4PFCHSL1FastL2L3ResidualCorrectorTask)
