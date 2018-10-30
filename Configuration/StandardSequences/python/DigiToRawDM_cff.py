import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.DigiToRaw_cff import *

# Re-define inputs to look at the DataMixer output
#
# In premixing stage2, need to use the original ones for muons
from Configuration.ProcessModifiers.premix_stage2_cff import premix_stage2
#
siPixelRawData.InputLabel = "mixData:siPixelDigisDM"
SiStripDigiToRaw.InputDigis = "mixData:siStripDigisDM"
#
ecalPacker.Label = 'DMEcalDigis'
ecalPacker.InstanceEB = 'ebDigis'
ecalPacker.InstanceEE = 'eeDigis'
ecalPacker.labelEBSRFlags = "DMEcalDigis:ebSrFlags"
ecalPacker.labelEESRFlags = "DMEcalDigis:eeSrFlags"
ecalPacker.labelTT = 'DMEcalTriggerPrimitiveDigis'
esDigiToRaw.Label = 'DMEcalPreshowerDigis'
#
hcalRawDataVME.HBHE = "DMHcalDigis"
hcalRawDataVME.HF = "DMHcalDigis"
hcalRawDataVME.HO = "DMHcalDigis"
hcalRawDataVME.ZDC = "mixData"
hcalRawDataVME.TRIG = "DMHcalTriggerPrimitiveDigis"
#
(~premix_stage2).toModify(cscpacker,
    wireDigiTag = "mixData:MuonCSCWireDigisDM",
    stripDigiTag = "mixData:MuonCSCStripDigisDM",
    comparatorDigiTag = "mixData:MuonCSCComparatorDigisDM"
)
(~premix_stage2).toModify(dtpacker, digiColl = 'mixData')
(~premix_stage2).toModify(rpcpacker, InputLabel = "mixData")

DigiToRaw.remove(castorRawData)

#castorRawData.CASTOR = cms.untracked.InputTag("castorDigis")
#

from Configuration.Eras.Modifier_run2_HCAL_2017_cff import run2_HCAL_2017
run2_HCAL_2017.toModify( hcalRawDataVME,
    HBHE = "",
    HF = "",
    TRIG = "",
)
run2_HCAL_2017.toModify( hcalRawDatauHTR,
    HBHEqie8 = "DMHcalDigis",
    HFqie8 = "DMHcalDigis",
    QIE10 = "DMHcalDigis:HFQIE10DigiCollection",
    QIE11 = "DMHcalDigis:HBHEQIE11DigiCollection",
    TP = "DMHcalTriggerPrimitiveDigis",
)


if 'caloLayer1RawFed1354' in globals():
    from Configuration.Eras.Modifier_stage2L1Trigger_cff import stage2L1Trigger

    stage2L1Trigger.toModify(caloLayer1RawFed1354,
                             ecalDigis= "DMEcalTriggerPrimitiveDigis",
                             hcalDigis= "DMHcalTriggerPrimitiveDigis"
                             )
    stage2L1Trigger.toModify(caloLayer1RawFed1356,
                             ecalDigis= "DMEcalTriggerPrimitiveDigis",
                             hcalDigis= "DMHcalTriggerPrimitiveDigis"
                             )
    stage2L1Trigger.toModify(caloLayer1RawFed1358,
                             ecalDigis= "DMEcalTriggerPrimitiveDigis",
                             hcalDigis= "DMHcalTriggerPrimitiveDigis"
                             )
