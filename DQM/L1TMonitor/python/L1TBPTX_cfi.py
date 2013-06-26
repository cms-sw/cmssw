import FWCore.ParameterSet.Config as cms

l1tBPTX = cms.EDAnalyzer("L1TBPTX",

  dqmStore                = cms.untracked.bool(True),
  disableROOToutput       = cms.untracked.bool(True),
  verbose                 = cms.untracked.bool(False),
  inputTagScalersResults  = cms.InputTag("scalersRawToDigi"),
  inputTagL1GtDataDaq     = cms.InputTag("gtDigis"),
  inputTagtEvmSource      = cms.InputTag("gtEvmDigis"),

  # Online
  oracleDB   = cms.string("oracle://CMS_OMDS_LB/CMS_TRG_R"),
  pathCondDB = cms.string("/nfshome0/centraltspro/secure/"),

  # Offline
  #oracleDB   = cms.string("oracle://cms_orcon_adg/CMS_COND_31X_L1T"),
  #pathCondDB = cms.string("/afs/cern.ch/cms/DB/conddb"),

  # Categories to process
  MonitorBits = cms.VPSet(
    cms.PSet(
      testName  = cms.string('Tech_BPTX_AND'),
      bitType   = cms.bool(False), #True: Algo, False:. Tech
      bitNumber = cms.int32(0),
      bitOffset = cms.int32(0),
    ),
    cms.PSet(
      testName  = cms.string('Tech_preBPTX_Veto'),
      bitType   = cms.bool(False), #True: Algo, False:. Tech
      bitNumber = cms.int32(16),
      bitOffset = cms.int32(-1),
    ),
  ),
  
  MonitorRates = cms.VPSet(
      cms.PSet(
      testName  = cms.string('Algo_BPTX_AND'),
      bitType   = cms.bool(True), #True: Algo, False:. Tech
      bitNumber = cms.int32(0),
    ),
  ),
  
)
