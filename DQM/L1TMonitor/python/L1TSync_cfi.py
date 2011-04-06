import FWCore.ParameterSet.Config as cms

l1tSync = cms.EDAnalyzer("L1TSync",

  dqmStore                = cms.untracked.bool(True),
  disableROOToutput       = cms.untracked.bool(True),
  verbose                 = cms.untracked.bool(False),
  inputTagScalersResults  = cms.InputTag("scalersRawToDigi","","DQM"),
  inputTagL1GtDataDaq     = cms.InputTag("gtDigis"),

  # Index for the prescale set to be used 
  # as reference
  refPrescaleSet = cms.int32(0), 

  # Categories to process
  categories = cms.PSet(
    cms.PSet(
      Mu     = cms.untracked.bool(True),
      EG     = cms.untracked.bool(True),
      IsoEG  = cms.untracked.bool(True),
      Jet    = cms.untracked.bool(True),
      CenJet = cms.untracked.bool(True),
      ForJet = cms.untracked.bool(True),
      TauJet = cms.untracked.bool(True),
      ETM    = cms.untracked.bool(True),
      ETT    = cms.untracked.bool(True),
      HTT    = cms.untracked.bool(True),
      HTM    = cms.untracked.bool(True),
    ),
  ),

)




