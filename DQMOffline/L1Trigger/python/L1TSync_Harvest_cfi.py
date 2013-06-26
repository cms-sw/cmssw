import FWCore.ParameterSet.Config as cms

l1tSync_Harvest = cms.EDAnalyzer("L1TSync_Harvest",

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
  #oracleDB   = cms.string("oracle://cms_orcoff_prod/CMS_COND_31X_L1T")
  #pathCondDB = cms.string("/afs/cern.ch/cms/DB/conddb"),                

  # Index for the prescale set to be used 
  # as reference
  refPrescaleSet = cms.int32(0), 

  # Categories to process
  Categories = cms.PSet(

      # Global parameters for algo selection
      forceGlobalParameters = cms.bool(False),  # Force use of global over bit-by-bit parameters 
      doGlobalAutoSelection = cms.bool(False),  # Do automatic/fixed algo selection for all monitored algos

      BPTX = cms.PSet(
        monitor       = cms.bool(True),
        algo          = cms.string("L1_ZeroBias"),
        CertMinEvents = cms.int32(50),
      ),
      Mu = cms.PSet(
        monitor         = cms.bool(True),
        doAutoSelection = cms.bool(True),
        algo            = cms.string(""),
        CertMinEvents   = cms.int32(20),
      ),
      EG = cms.PSet(
        monitor         = cms.bool(True),
        doAutoSelection = cms.bool(True),
        algo            = cms.string(""),
        CertMinEvents   = cms.int32(20),
      ),
      IsoEG = cms.PSet( 
        monitor         = cms.bool(True),
        doAutoSelection = cms.bool(True),
        algo            = cms.string(""),
        CertMinEvents   = cms.int32(20),
      ),
      Jet = cms.PSet(
        monitor         = cms.bool(True),
        doAutoSelection = cms.bool(True),
        algo            = cms.string(""),
        CertMinEvents   = cms.int32(20),
      ),
      CenJet = cms.PSet(
        monitor         = cms.bool(False),
        doAutoSelection = cms.bool(True),
        algo            = cms.string(""),
        CertMinEvents   = cms.int32(50),
      ),
      ForJet = cms.PSet(
        monitor         = cms.bool(False),
        doAutoSelection = cms.bool(True),
        algo            = cms.string(""),
        CertMinEvents   = cms.int32(20),
      ),
      TauJet = cms.PSet(
        monitor         = cms.bool(False),
        doAutoSelection = cms.bool(True),
        algo            = cms.string(""),
        CertMinEvents   = cms.int32(20),
      ),
      ETM = cms.PSet(
        monitor         = cms.bool(True),
        doAutoSelection = cms.bool(True),
        algo            = cms.string(""),
        CertMinEvents   = cms.int32(20),
      ),
      ETT = cms.PSet(   
        monitor         = cms.bool(True),
        doAutoSelection = cms.bool(True),
        algo            = cms.string(""),
        CertMinEvents   = cms.int32(20),
      ),
      HTT = cms.PSet(   
        monitor         = cms.bool(True),
        doAutoSelection = cms.bool(True),
        algo            = cms.string(""),
        CertMinEvents   = cms.int32(20),
      ),
      HTM = cms.PSet(
        monitor         = cms.bool(True),   
        doAutoSelection = cms.bool(True),
        algo            = cms.string(""),
        CertMinEvents   = cms.int32(20),
      ),
  ),


)
