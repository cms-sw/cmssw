import FWCore.ParameterSet.Config as cms

l1tSync = cms.EDAnalyzer("L1TSync",

  dqmStore                = cms.untracked.bool(True),
  disableROOToutput       = cms.untracked.bool(True),
  verbose                 = cms.untracked.bool(False),
  inputTagScalersResults  = cms.InputTag("scalersRawToDigi","","DQM"),
  inputTagL1GtDataDaq     = cms.InputTag("gtDigis"),
 
  # Online
  oracleDB   = cms.string("oracle://CMS_OMDS_LB/CMS_TRG_R"),
  pathCondDB = cms.string("/nfshome0/centraltspro/secure/"),                

  # Offline
  #oracleDB   = cms.string("oracle://cms_orcoff_prod/CMS_COND_31X_L1T"), # For offline
  #pathCondDB = cms.string("/afs/cern.ch/cms/DB/conddb"),                

  # Index for the prescale set to be used 
  # as reference
  refPrescaleSet = cms.int32(0), 

  # Categories to process
  Categories = cms.PSet(

      # Global parameters for algo selection
      forceGlobalParameters = cms.bool(False),  # Force use of global over bit-by-bit parameters 
      doGlobalAutoSelection = cms.bool(False),  # Do automatic/fixed algo selection for all monitored algos

      Mu = cms.PSet(
        monitor         = cms.bool(True),
        doAutoSelection = cms.bool(True),
        algo            = cms.string(""),
      ),
      EG = cms.PSet(
        monitor         = cms.bool(True),
        doAutoSelection = cms.bool(True),
        algo            = cms.string("L1_SingleEG8"),
      ),
      IsoEG = cms.PSet( 
        monitor         = cms.bool(True),
        doAutoSelection = cms.bool(True),
        algo            = cms.string(""),
      ),
      Jet = cms.PSet(
        monitor         = cms.bool(True),
        doAutoSelection = cms.bool(True),
        algo            = cms.string(""),
      ),
      CenJet = cms.PSet(
        monitor         = cms.bool(True),
        doAutoSelection = cms.bool(True),
        algo            = cms.string(""),
      ),
      ForJet = cms.PSet(
        monitor         = cms.bool(True),
        doAutoSelection = cms.bool(True),
        algo            = cms.string(""),
      ),
      TauJet = cms.PSet(
        monitor         = cms.bool(True),
        doAutoSelection = cms.bool(True),
        algo            = cms.string(""),
      ),
      ETM = cms.PSet(
        monitor         = cms.bool(True),
        doAutoSelection = cms.bool(True),
        algo            = cms.string(""),
      ),
      ETT = cms.PSet(   
        monitor         = cms.bool(True),
        doAutoSelection = cms.bool(True),
        algo            = cms.string(""),
      ),
      HTT = cms.PSet(   
        monitor         = cms.bool(True),
        doAutoSelection = cms.bool(True),
        algo            = cms.string(""),
      ),
      HTM = cms.PSet(
        monitor         = cms.bool(True),   
        doAutoSelection = cms.bool(True),
        algo            = cms.string(""),
      ),
  ),
)

