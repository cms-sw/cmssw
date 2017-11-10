import FWCore.ParameterSet.Config as cms

muonGEMDigis = cms.EDProducer("GEMRawToDigiModule",
    InputLabel = cms.InputTag("rawDataCollector"),
<<<<<<< HEAD:EventFilter/GEMRawToDigi/python/gemRawToDigi_cfi.py
    useDBEMap = cms.bool(True),    
=======
    UnpackStatusDigis = cms.bool(False),
    useDBEMap = cms.bool(False),
>>>>>>> adding packing and unpacking to std seq:EventFilter/GEMRawToDigi/python/gemUnpacker_cfi.py
)
