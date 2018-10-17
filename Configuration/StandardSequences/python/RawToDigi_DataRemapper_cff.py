import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.RawToDigi_cff import *

RawToDigi.insert(0, rawDataRemapperByLabel)
RawToDigi_noTk.insert(0, rawDataRemapperByLabel)
RawToDigi_pixelOnly.insert(0, rawDataRemapperByLabel)

ecalDigis.DoRegional = False
#False by default ecalDigis.DoRegional = False

