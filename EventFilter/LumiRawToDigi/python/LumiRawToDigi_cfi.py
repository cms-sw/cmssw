import FWCore.ParameterSet.Config as cms
import EventFilter.LumiRawToDigi.lumiRawToDigi_cfi

lumiDigis = EventFilter.LumiRawToDigi.lumiRawToDigi_cfi.lumiRawToDigi.clone()
lumiDigis.InputLabel = cms.InputTag("rawDataCollector")
