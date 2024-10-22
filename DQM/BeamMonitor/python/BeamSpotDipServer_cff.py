import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer

beamSpotDipServer = DQMEDAnalyzer("BeamSpotDipServer",
  monitorName = cms.untracked.string("BeamSpotDipServer"),
  #
  verbose = cms.untracked.bool(False),
  testing = cms.untracked.bool(False),
  #
  subjectCMS = cms.untracked.string("dip/CMS/Tracker/BeamSpot"),
  subjectLHC = cms.untracked.string("dip/CMS/LHC/LuminousRegion"),
  subjectPV  = cms.untracked.string("dip/CMS/Tracker/PrimaryVertices"),
  #
  readFromNFS = cms.untracked.bool(True),
  #
  dcsRecordInputTag = cms.untracked.InputTag ( "onlineMetaDataDigis" ),
  #
  sourceFile  = cms.untracked.string(
    "/nfshome0/dqmpro/BeamMonitorDQM/BeamFitResults.txt"),
  sourceFile1 = cms.untracked.string(
    "/nfshome0/dqmpro/BeamMonitorDQM/BeamFitResults_TkStatus.txt"),
  #
  timeoutLS = cms.untracked.vint32(1,2)
)

