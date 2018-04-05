import FWCore.ParameterSet.Config as cms

process = cms.Process("DUMP")
process.load("CondCore.CondDB.CondDB_cfi")
process.load("Configuration.Geometry.GeometryExtended2016dev_cff")
process.load("Configuration.Geometry.GeometryExtended2016devReco_cff")


process.maxEvents = cms.untracked.PSet(  input = cms.untracked.int32(1) )

process.source = cms.Source("EmptySource",
                            numberEventsInRun = cms.untracked.uint32(1),
                            firstRun = cms.untracked.uint32(1)
                            )

process.hcal_db_producer = cms.ESProducer("HcalDbProducer",
                                          dump = cms.untracked.vstring(''),
                                          file = cms.untracked.string('')
                                          )


#--------------------- replacing conditions, fileinpath starts from /src 
process.es_ascii = cms.ESSource("HcalTextCalibrations",
                                input = cms.VPSet(
        cms.PSet(
	    object = cms.string('FrontEndMap'),
	    file = cms.FileInPath('HCALmap_forNoiseStudy_2016_NoQIE10.txt')
            )
        )
                                )

process.es_prefer = cms.ESPrefer('HcalTextCalibrations','es_ascii')


process.dumpcond = cms.EDAnalyzer("HcalDumpConditions",
                                  dump = cms.untracked.vstring("FrontEndMap")
                                  )

process.p = cms.Path(process.dumpcond)
