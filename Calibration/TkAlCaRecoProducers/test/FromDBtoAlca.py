import FWCore.ParameterSet.Config as cms


process = cms.Process("readDB")
process.load("FWCore.MessageLogger.MessageLogger_cfi")


process.load("CondCore.DBCommon.CondDBSetup_cfi")



process.BeamSpotDBSource = cms.ESSource("PoolDBESSource",
                                        process.CondDBSetup,
                                        toGet = cms.VPSet(cms.PSet(
    record = cms.string('BeamSpotObjectsRcd'),
    tag = cms.string('BeamSpotObjects_2009_LumiBased_SigmaZ_v14_offline')
    )),
                                        connect = cms.string('frontier://cmsfrontier.cern.ch:8000/Frontier/CMS_COND_31X_BEAMSPOT')
                                        #connect = cms.string('sqlite_file:Early900GeVCollision_7p4cm_STARTUP_mc.db')
                                        #connect = cms.string('oracle://cms_orcoff_prod/CMS_COND_31X_BEAMSPOT')
                                        #connect = cms.string('frontier://PromptProd/CMS_COND_31X_BEAMSPOT')
                                        )

process.source = cms.Source("EmptySource",
                            processingMode=cms.untracked.string('RunsLumisAndEvents'),
                            firstRun = cms.untracked.uint32(132601),
                            lastRun = cms.untracked.uint32(138751),
                            numberEventsInLuminosityBlock = cms.untracked.uint32(1),
                            numberEventsInRun=cms.untracked.uint32(10),
                            firstLuminosityBlock = cms.untracked.uint32(1)                           
                            )

process.out = cms.OutputModule( "PoolOutputModule",
                                fileName = cms.untracked.string( '/tmp/yumiceva/AlcaBeamSpotDB.root' ),
                                outputCommands = cms.untracked.vstring("keep *")
                                )

process.maxEvents = cms.untracked.PSet(
     input = cms.untracked.int32(100)
)

process.maxLuminosityBlocks=cms.untracked.PSet(
     input=cms.untracked.int32(100)
)
 
process.beamspot = cms.EDProducer("AlcaBeamSpotFromDB")


process.p = cms.Path(process.beamspot)

process.e = cms.EndPath( process.out )

