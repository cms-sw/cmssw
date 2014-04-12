import FWCore.ParameterSet.Config as cms

EcalRegionCablingESProducer = cms.ESProducer("EcalRegionCablingESProducer",
                                             esMapping = cms.PSet(LookupTable = cms.FileInPath("EventFilter/ESDigiToRaw/data/ES_lookup_table.dat"))
                                             )


