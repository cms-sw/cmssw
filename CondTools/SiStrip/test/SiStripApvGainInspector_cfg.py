import FWCore.ParameterSet.Config as cms

process = cms.Process("Demo")
process.load("FWCore.MessageService.MessageLogger_cfi")

process.source = cms.Source("EmptyIOVSource",
                            firstValue = cms.uint64(317340),
                            lastValue  = cms.uint64(317340),
                            timetype  = cms.string('runnumber'),
                            interval = cms.uint64(1)
                            )


process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1) )

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, '101X_dataRun2_Express_v7', '')

process.load("Configuration.Geometry.GeometryRecoDB_cff")

####################################################################
# Output file
####################################################################
process.TFileService = cms.Service("TFileService",
                                   fileName=cms.string("APVGainsTree.root")
                                   )                                    


##
## Output database (in this case local sqlite file)
##
process.load("CondCore.CondDB.CondDB_cfi")
process.CondDB.connect = "sqlite_file:updatedGains.db"
process.PoolDBOutputService = cms.Service("PoolDBOutputService",
                                          process.CondDB,
                                          timetype = cms.untracked.string('runnumber'),
                                          toPut = cms.VPSet(cms.PSet(record = cms.string('SiStripApvGainRcd'),
                                                                     tag = cms.string('modifiedGains'))
                                          ))

process.demo = cms.EDAnalyzer('SiStripApvGainInspector',
                              fitMode = cms.int32(2), # landau around max
                              #inputFile = cms.untracked.string("DQM_V0001_R000999999__StreamExpress__Run2018B-PromptCalibProdSiStripGainsAAG-Express-v1-317279-317340__ALCAPROMPT.root"),
                              inputFile = cms.untracked.string("root://eoscms.cern.ch//eos/cms/store/group/alca_global/multiruns/results/prod//slc6_amd64_gcc630/CMSSW_10_1_5/86791_1p_0f/DQM_V0001_R000999999__StreamExpress__Run2018B-PromptCalibProdSiStripGainsAAG-Express-v1-317382-317488__ALCAPROMPT.root"),
                              ### FED 387
                              selectedModules = cms.untracked.vuint32(436281608,436281604,436281592,436281624,436281620,436281644,436281640,436281648,436281668,436281680,436281684,436281688,436281720,436281700,436281708,436281556,436281552,436281704,436281764,436281768,436281572,436281576,436281748,436281744,436281740,436281780,436281784,436281612,436281616,436281588,436281580,436281584,436281636,436281656,436281652,436281676,436281672,436281732,436281736,436281716,436281712,436281776,436281772,436281548,436281544,436281540,436281752,436281560)
                              #### FED 434
                              ###selectedModules = cms.untracked.vuint32(436266168,436266028,436266020,436266024,436266160,436266164,436266000,436266004,436266008,436265976,436265972,436266064,436266060,436266068,436265964,436265960,436265968,436265988,436266088,436266084,436266040,436266128,436266116,436266132,436266136,436266156,436266152,436266100,436266032,436266036,436266096,436266052,436266056,436265956,436266092,436265992,436265996,436266104,436266072,436266124,436266120,436266148)
                              )

process.p = cms.Path(process.demo)
