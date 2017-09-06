import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.DigiToRaw_cff import *
#
# Re-define inputs to look at the DataMixer output
#
siPixelRawData.InputLabel = cms.InputTag("mixData:siPixelDigisDM")
SiStripDigiToRaw.InputModuleLabel = cms.string('mixData')
SiStripDigiToRaw.InputDigiLabel = cms.string('siStripDigisDM')
#
ecalPacker.Label = 'DMEcalDigis'
ecalPacker.InstanceEB = 'ebDigis'
ecalPacker.InstanceEE = 'eeDigis'
ecalPacker.labelEBSRFlags = "DMEcalDigis:ebSrFlags"
ecalPacker.labelEESRFlags = "DMEcalDigis:eeSrFlags"
ecalPacker.labelTT = cms.InputTag('DMEcalTriggerPrimitiveDigis')
esDigiToRaw.Label = cms.string('DMEcalPreshowerDigis')
#
hcalRawDataVME.HBHE = cms.untracked.InputTag("DMHcalDigis")
hcalRawDataVME.HF = cms.untracked.InputTag("DMHcalDigis")
hcalRawDataVME.HO = cms.untracked.InputTag("DMHcalDigis") 
hcalRawDataVME.ZDC = cms.untracked.InputTag("mixData")
hcalRawDataVME.TRIG = cms.untracked.InputTag("DMHcalTriggerPrimitiveDigis")
#
cscpacker.wireDigiTag = cms.InputTag("mixData","MuonCSCWireDigisDM")
cscpacker.stripDigiTag = cms.InputTag("mixData","MuonCSCStripDigisDM")
cscpacker.comparatorDigiTag = cms.InputTag("mixData","MuonCSCComparatorDigisDM")
dtpacker.digiColl = cms.InputTag('mixData')
#dtpacker.digiColl = cms.InputTag('simMuonDTDigis')
rpcpacker.InputLabel = cms.InputTag("mixData")

DigiToRaw.remove(castorRawData)

#castorRawData.CASTOR = cms.untracked.InputTag("castorDigis")
#

from Configuration.Eras.Modifier_run2_HCAL_2017_cff import run2_HCAL_2017
run2_HCAL_2017.toModify( hcalRawDataVME,
    HBHE = cms.untracked.InputTag(""),
    HF = cms.untracked.InputTag(""),
    TRIG = cms.untracked.InputTag(""),
)
run2_HCAL_2017.toModify( hcalRawDatauHTR,
    HBHEqie8 = cms.InputTag("DMHcalDigis"),
    HFqie8 = cms.InputTag("DMHcalDigis"),
    QIE10 = cms.InputTag("DMHcalDigis","HFQIE10DigiCollection"),
    QIE11 = cms.InputTag("DMHcalDigis","HBHEQIE11DigiCollection"),
    TP = cms.InputTag("DMHcalTriggerPrimitiveDigis"),
)

