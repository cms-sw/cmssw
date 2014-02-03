import FWCore.ParameterSet.Config as cms

from DQMOffline.L1Trigger.L1TRateOfflineParams_cff import RateParams

l1tRate_Offline = cms.EDAnalyzer("L1TRate_Offline",

  #-------------------------------------------------------
  #-------------------- ATTENTION-------------------------
  #-------------------------------------------------------
  # The parameter lsShiftGTRates shifts the LS number for 
  # the GT Rates taken from SCAL by a value defined by the
  # user. Right now it is set to -1 to compensate a bug 
  # in SCAL that is described in:
  # https://savannah.cern.ch/support/?122368
  # As soon as this bug is corrected this value MUST be
  # set to 0 again.
  lsShiftGTRates             = cms.untracked.int32(-1),

  verbose                    = cms.untracked.bool(False),
  dqmStore                   = cms.untracked.bool(True),
  disableROOToutput          = cms.untracked.bool(True),
  inputTagScalersResults     = cms.InputTag("scalersRawToDigi"),
  inputTagL1GtDataDaq        = cms.InputTag("gtDigis"),
  useHFDeadTimeNormalization = cms.untracked.bool(False),
  
  # Plot Parameters
  minInstantLuminosity = cms.double (100),
  maxInstantLuminosity = cms.double(10000),
  
  # Index for the prescale set to be used as reference
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
      CenJet = cms.untracked.bool(False), # Currently there is no unmasked trigger in this category
      ForJet = cms.untracked.bool(False), # Currently there is no unmasked trigger in this category
      TauJet = cms.untracked.bool(False), # Currently there is no unmasked trigger in this category
      ETM    = cms.untracked.bool(True),
      ETT    = cms.untracked.bool(True),
      HTT    = cms.untracked.bool(True),
      HTM    = cms.untracked.bool(True),
    ),
  ),

  # Algo XSec Fits  
  # srcAlgoXSecFit = 0 -> From WbM via OMDS
  # srcAlgoXSecFit = 1 -> From python
  srcAlgoXSecFit = cms.int32(1),

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
