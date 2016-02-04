import FWCore.ParameterSet.Config as cms

ESUnpackerWorkerESProducer = cms.ESProducer("ESUnpackerWorkerESProducer",
                                            ComponentName = cms.string('esRawToRecHit'),
                                            DCCDataUnpacker = cms.PSet(
                                                LookupTable = cms.untracked.FileInPath("EventFilter/ESDigiToRaw/data/ES_lookup_table.dat")
                                            ),
                                            RHAlgo = cms.PSet(
                                                Type = cms.string('ESRecHitWorker')
                                            )
                                            )
                                           
