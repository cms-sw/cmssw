import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.RawToDigi_cff import *

from EventFilter.EcalRawToDigi.EcalRawDataRecovery_cfi import ecalRawDataRecovery

RawToDigi = cms.Sequence(
                         siPixelDigis
                         +siStripDigis
                         +(ecalRawDataRecovery*ecalDigis)
                         +ecalPreshowerDigis
                         +hcalDigis
                         +muonCSCDigis
                         +muonDTDigis
                         +muonRPCDigis
                         +castorDigis
                         +scalersRawToDigi
                         +tcdsDigis
                         +L1TRawToDigi)

ecalDigis.DoRegional = False

#set those back to "source"
#False by default ecalDigis.DoRegional = False

