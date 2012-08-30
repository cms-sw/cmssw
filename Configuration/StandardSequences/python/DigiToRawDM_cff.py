import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.DigiToRaw_cff import *
#
# Re-define inputs to look at the DataMixer output
#
siPixelRawData.InputLabel = cms.InputTag("mixData:siPixelDigisDM")
SiStripDigiToRaw.InputModuleLabel = cms.string('mixData')
SiStripDigiToRaw.InputDigiLabel = cms.string('siStripDigisDM')
#
ecalPacker.Label = 'mixData'
ecalPacker.InstanceEB = 'EBDigiCollectionDM'
ecalPacker.InstanceEE = 'EEDigiCollectionDM'
ecalPacker.labelEBSRFlags = "DMEcalDigis:ebSrFlags"
ecalPacker.labelEESRFlags = "DMEcalDigis:eeSrFlags"
ecalPacker.labelTT = cms.InputTag('DMEcalTriggerPrimitiveDigis')
esDigiToRaw.Label = cms.string('DMEcalPreshowerDigis')
#
hcalRawData.HBHE = cms.untracked.InputTag("DMHcalDigis")
hcalRawData.HF = cms.untracked.InputTag("DMHcalDigis")
hcalRawData.HO = cms.untracked.InputTag("DMHcalDigis") 
hcalRawData.ZDC = cms.untracked.InputTag("mixData")
#
cscpacker.wireDigiTag = cms.InputTag("mixData","MuonCSCWireDigisDM")
cscpacker.stripDigiTag = cms.InputTag("mixData","MuonCSCStripDigisDM")
cscpacker.comparatorDigiTag = cms.InputTag("mixData","MuonCSCComparatorDigisDM")
dtpacker.digiColl = cms.InputTag('mixData')
#dtpacker.digiColl = cms.InputTag('simMuonDTDigis')
rpcpacker.InputLabel = cms.InputTag("mixData")
castorRawData.CASTOR = cms.untracked.InputTag("castorDigis")
#
