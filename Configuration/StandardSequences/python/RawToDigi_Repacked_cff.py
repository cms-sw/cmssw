import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.RawToDigi_cff import *

scalersRawToDigi.scalersInputTag = 'rawDataRepacker'
csctfDigis.producer = 'rawDataRepacker'
dttfDigis.DTTF_FED_Source = 'rawDataRepacker'
gctDigis.inputLabel = 'rawDataRepacker'
gtDigis.DaqGtInputTag = 'rawDataRepacker'
gtEvmDigis.EvmGtInputTag = 'rawDataRepacker'
siPixelDigis.InputLabel = 'rawDataRepacker'
siStripDigis.ProductLabel = 'rawDataRepacker'
#False by default ecalDigis.DoRegional = False
ecalDigis.InputLabel = 'rawDataRepacker'
ecalPreshowerDigis.sourceTag = 'rawDataRepacker'
hcalDigis.InputLabel = 'rawDataRepacker'
muonCSCDigis.InputObjects = 'rawDataRepacker'
muonDTDigis.inputLabel = 'rawDataRepacker'
muonRPCDigis.InputLabel = 'rawDataRepacker'
castorDigis.InputLabel = 'rawDataRepacker'

siStripVRDigis = siStripDigis.clone(ProductLabel = 'virginRawDataRepacker')
RawToDigi_withVR = cms.Sequence(RawToDigi + siStripVRDigis)
