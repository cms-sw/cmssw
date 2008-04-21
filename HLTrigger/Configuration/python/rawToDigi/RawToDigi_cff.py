import FWCore.ParameterSet.Config as cms

from HLTrigger.Configuration.rawToDigi.SetupRawToDigi_cff import *
from HLTrigger.Configuration.rawToDigi.EcalRecord_cff import *
from HLTrigger.Configuration.rawToDigi.SiPixelRawToDigi_cff import *
from HLTrigger.Configuration.rawToDigi.SiStripRawToClusters_cff import *
from HLTrigger.Configuration.rawToDigi.EcalRawToDigi_cff import *
from HLTrigger.Configuration.rawToDigi.EcalESRawToDigi_cff import *
from HLTrigger.Configuration.rawToDigi.HcalRawToDigi_cff import *
from HLTrigger.Configuration.rawToDigi.CSCRawToDigi_cff import *
from HLTrigger.Configuration.rawToDigi.DTRawToDigi_cff import *
from HLTrigger.Configuration.rawToDigi.RPCRawToDigi_cff import *
RawToDigi = cms.Sequence(SiPixelRawToDigi+SiStripRawToClusters+EcalRawToDigi+EcalESRawToDigi+HcalRawToDigi+CSCRawToDigi+DTRawToDigi+RPCRawToDigi)

