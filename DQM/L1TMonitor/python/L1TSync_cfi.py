import FWCore.ParameterSet.Config as cms

from HLTrigger.HLTfilters.hltHighLevel_cfi import *

l1tSyncFilter = hltHighLevel.clone(TriggerResultsTag ="TriggerResults::HLT")
l1tSyncFilter.throw = cms.bool(False)
l1tSyncFilter.HLTPaths = ['HLT_ZeroBias_v*',
                          'HLT_L1ETM30_v*',
                          'HLT_L1MultiJet_v*',
                          'HLT_L1SingleEG12_v',
                          'HLT_L1SingleEG5_v*',
                          'HLT_L1SingleJet16_v*',
                          'HLT_L1SingleJet36_v*',
                          'HLT_L1SingleMu10_v*',
                          'HLT_L1SingleMu20_v*']



l1tSyncMonitor = cms.EDAnalyzer("L1TSync",

  dqmStore                = cms.untracked.bool(True),
  disableROOToutput       = cms.untracked.bool(True),
  verbose                 = cms.untracked.bool(False),
  inputTagScalersResults  = cms.InputTag("scalersRawToDigi","","DQM"),
  inputTagL1GtDataDaq     = cms.InputTag("gtDigis"),
  inputTagtEvmSource      = cms.InputTag("gtEvmDigis","","DQM"),

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

      BPTX = cms.PSet(
        monitor = cms.bool(True),
        algo    = cms.string("L1_ZeroBias"),
      ),
      Mu = cms.PSet(
        monitor         = cms.bool(True),
        doAutoSelection = cms.bool(True),
        algo            = cms.string(""),
      ),
      EG = cms.PSet(
        monitor         = cms.bool(True),
        doAutoSelection = cms.bool(True),
        algo            = cms.string(""),
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
        monitor         = cms.bool(False),
        doAutoSelection = cms.bool(True),
        algo            = cms.string(""),
      ),
      ForJet = cms.PSet(
        monitor         = cms.bool(False),
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

l1tSync = cms.Sequence(l1tSyncFilter*l1tSyncMonitor)
