import FWCore.ParameterSet.Config as cms

from DQM.L1TMonitor.L1TRateParams_cff import RateParams

l1tRate = cms.EDAnalyzer("L1TRate",

  dqmStore               = cms.untracked.bool(True),
  disableROOToutput      = cms.untracked.bool(True),
  verbose                = cms.untracked.bool(False),
  inputTagScalersResults = cms.InputTag("scalersRawToDigi","","DQM"),
  inputTagL1GtDataDaq    = cms.InputTag("gtDigis"),

  # Index for the prescale set to be used 
  # as reference
  refPrescaleSet = cms.int32(0), 

  # Test if scalLS==eventLS-1  
  testEventScalLS = cms.untracked.bool(True), # True for grid jobs

  # Categories to process
  categories = cms.PSet(
    cms.PSet(
      Mu     = cms.untracked.bool(True),
      EG     = cms.untracked.bool(True),
      IsoEG  = cms.untracked.bool(True),
      Jet    = cms.untracked.bool(True),
      CenJet = cms.untracked.bool(False),
      ForJet = cms.untracked.bool(False),
      TauJet = cms.untracked.bool(True),
      ETM    = cms.untracked.bool(True),
      ETT    = cms.untracked.bool(True),
      HTT    = cms.untracked.bool(True),
      HTM    = cms.untracked.bool(True),
    ),
  ),

  # Algo XSec Fits  
  # srcAlgoXSecFit = 0 -> From WbM via OMDS
  # srcAlgoXSecFit = 1 -> From python
  srcAlgoXSecFit = cms.int32(0),

  # if srcAlgoXSecFit = 0 we need to define 
  ## Online
  oracleDB   = cms.string("oracle://CMS_OMDS_LB/CMS_TRG_R"),
  pathCondDB = cms.string("/nfshome0/centraltspro/secure/"),                

  ## Offline
  #oracleDB   = cms.string("oracle://cms_orcoff_prod/CMS_COND_31X_L1T"), # For offline
  #pathCondDB = cms.string("/afs/cern.ch/cms/DB/conddb"), 

  # if srcAlgoXSecFit = 1 we need to define 
  fitParameters = RateParams

)
