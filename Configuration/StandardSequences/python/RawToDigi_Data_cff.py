import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.RawToDigi_cff import *

RawToDigi = cms.Sequence(
                         siPixelDigis
                         +siStripDigis
                         +ecalDigis
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

