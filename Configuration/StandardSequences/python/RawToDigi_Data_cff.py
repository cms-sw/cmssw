import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.RawToDigi_cff import *

RawToDigi = cms.Sequence(csctfDigis
                         +dttfDigis
                         +gctDigis
                         +gtDigis
                         +gtEvmDigis
                         +siPixelDigis
                         +siStripDigis
                         +ecalDigis
                         +ecalPreshowerDigis
                         +hcalDigis
                         +muonCSCDigis
                         +muonDTDigis
                         +muonRPCDigis
                         +castorDigis
                         +scalersRawToDigi
                         +tcdsDigis)

RawToDigi_woGCT = cms.Sequence(csctfDigis
                               +dttfDigis
                               +gtDigis
                               +gtEvmDigis
                               +siPixelDigis
                               +siStripDigis
                               +ecalDigis
                               +ecalPreshowerDigis
                               +hcalDigis
                               +muonCSCDigis
                               +muonDTDigis
                               +muonRPCDigis
                               +castorDigis
                               +scalersRawToDigi
                               +tcdsDigis)

ecalDigis.DoRegional = False

#set those back to "source"
#False by default ecalDigis.DoRegional = False


