# The following comments couldn't be translated into the new config version:

#  include "RecoMuon/TrackingTools/data/MuonServiceProxy.cff"
#  include "Geometry/CMSCommonData/data/cmsIdealGeometryXML.cff"
#  include "MagneticField/Engine/data/volumeBasedMagneticField.cfi"
#  include "Geometry/CommonDetUnit/data/globalTrackingGeometry.cfi"
#  include "RecoMuon/DetLayers/data/muonDetLayerGeometry.cfi"

# the clients
#  include "DQMOffline/Muon/data/trackResidualsTest.cfi"
#  module qTester = QualityTester{
#     untracked int32 prescaleFactor = 1
#     untracked FileInPath qtList = "DQMOffline/Muon/data/QualityTests.xml" 
#  }

import FWCore.ParameterSet.Config as cms

process = cms.Process("Reco")
process.load("CondCore.DBCommon.CondDBSetup_cfi")

#
# DQM
#
process.load("DQMServices.Core.DQM_cfg")

process.load("DQMServices.Components.MEtoEDMConverter_cfi")

# the task
process.load("DQMOffline.JetMET.jetMETDQMOfflineSourceCosmic_cff")
process.jetMETAnalyzer.OutputMEsInRootFile = cms.bool(True)
process.jetMETAnalyzer.OutputFileName = cms.string('jetMETMonitoring_cruzet98154.root')

# check # of bins
process.load("DQMServices.Components.DQMStoreStats_cfi")

# for igprof
process.IgProfService = cms.Service("IgProfService",
  reportFirstEvent            = cms.untracked.int32(0),
  reportEventInterval         = cms.untracked.int32(25),
  reportToFileAtPostEvent     = cms.untracked.string("| gzip -c > igdqm.%I.gz")
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
     '/store/data/CRUZET09/Calo/RECO/v1/000/098/154/EADF3BE3-BE4F-DE11-8BB8-000423D9870C.root'
    )
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32( 100 )
)
process.Timing = cms.Service("Timing")

process.MessageLogger = cms.Service("MessageLogger",
    debugModules = cms.untracked.vstring('jetMETAnalyzer'),
    cout = cms.untracked.PSet(
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        jetMETAnalyzer = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        noLineBreaks = cms.untracked.bool(True),
        DEBUG = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        #FwkJob = cms.untracked.PSet(
        #    limit = cms.untracked.int32(0)
        #),
        threshold = cms.untracked.string('DEBUG')
    ),
    categories = cms.untracked.vstring('jetMETAnalyzer'),
    destinations = cms.untracked.vstring('cout')
)

process.FEVT = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring('keep *_MEtoEDMConverter_*_*'),
    fileName = cms.untracked.string('reco-grumm.root')
)

process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True) ## default is false

)

process.p = cms.Path(process.jetMETDQMOfflineSourceCosmic * process.dqmStoreStats)
process.outpath = cms.EndPath(process.FEVT)
process.DQM.collectorHost = ''

