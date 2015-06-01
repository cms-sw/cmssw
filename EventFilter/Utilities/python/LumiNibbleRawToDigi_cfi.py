import FWCore.ParameterSet.Config as cms
import EventFilter.Utilities.lumiNibbleRawToDigi_cfi

lumiNibbleDigis = EventFilter.Utilities.lumiNibbleRawToDigi_cfi.lumiNibbleRawToDigi.clone()
lumiNibbleDigis.InputLabel = cms.InputTag("rawDataCollector")
