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
process = cms.Process("JetMETAnalysis")
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
#
#
# DQM
#
process.load("DQMServices.Core.DQM_cfg")

process.load("DQMServices.Components.MEtoEDMConverter_cfi")

# the task
process.load("DQMOffline.JetMET.jetMETAnalyzer_cfi")
process.jetMETAnalyze.OutputMEsInRootFile = cms.bool(True)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
#forMC:    
#    '/store/relval/CMSSW_3_1_0_pre9/RelValQCD_Pt_80_120/GEN-SIM-RECO/IDEAL_31X_v1/0006/0268E2E0-5D4E-DE11-90B3-001D09F253D4.root'
# data:
       '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_v1/000/056/591/F2F1483E-416F-DD11-A270-001617E30CE8.root',
       '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_v1/000/057/313/4214986A-196D-DD11-BED7-000423D992DC.root',
       '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_v1/000/057/381/0A45CF4D-336D-DD11-A6EA-000423D6CAF2.root',
       '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_v1/000/057/381/FE634B34-316D-DD11-BABF-000423D6B358.root',
       '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_v1/000/057/394/C6CC14EE-316D-DD11-A3E8-000423D986A8.root',
       '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_v1/000/057/400/84750050-416D-DD11-9230-000423D6B48C.root',
       '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_v1/000/057/404/144C1DEA-466D-DD11-9A1F-000423D9880C.root',
       '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_v1/000/057/404/16ED5E7C-486D-DD11-A428-000423D6C8EE.root',
       '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_v1/000/057/404/2403B5E5-466D-DD11-B0F3-001617DBD332.root',
       '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_v1/000/057/404/3A9E9CB3-496D-DD11-A88D-001617E30CE8.root',
       '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_v1/000/057/404/3CCFE9B5-496D-DD11-B645-001617C3B778.root',
       '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_v1/000/057/404/564598B3-496D-DD11-8403-001617C3B76E.root',
       '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_v1/000/057/404/6639F8FF-486D-DD11-B710-000423D6CA6E.root',
       '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_v1/000/057/404/6E27F1B5-496D-DD11-8265-001617C3B5F4.root',
       '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_v1/000/057/404/76CF2D2C-4B6D-DD11-B626-000423D98DB4.root',
       '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_v1/000/057/404/92C32971-4A6D-DD11-8228-000423D6CAF2.root',
       '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_v1/000/057/404/9AC0042C-4B6D-DD11-B76D-000423D6CA02.root',
       '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_v1/000/057/404/A4DC012C-4B6D-DD11-B253-000423D98804.root',
       '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_v1/000/057/404/BA182670-4A6D-DD11-92AF-000423D6B358.root',
       '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_v1/000/057/404/BE0BFF2B-4B6D-DD11-BDE4-000423D9870C.root',
       '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_v1/000/057/404/C6CDF42A-4B6D-DD11-8E20-000423D6B42C.root',
       '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_v1/000/057/404/CC78012C-4B6D-DD11-A2A6-000423D6A6F4.root',
       '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_v1/000/057/404/D48991CD-4B6D-DD11-B5AB-000423D6BA18.root',
       '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_v1/000/057/404/D69E2487-4C6D-DD11-9391-000423D6AF24.root'
    )
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32( 10 )
)
process.Timing = cms.Service("Timing")

## process.MessageLogger = cms.Service("MessageLogger",
##     debugModules = cms.untracked.vstring('jetMETAnalyzer'),
##     cout = cms.untracked.PSet(
##         default = cms.untracked.PSet(
##             limit = cms.untracked.int32(0)
##         ),
##         jetMETAnalyzer = cms.untracked.PSet(
##             limit = cms.untracked.int32(10000000)
##         ),
##         noLineBreaks = cms.untracked.bool(True),
##         DEBUG = cms.untracked.PSet(
##             limit = cms.untracked.int32(0)
##         ),
##         FwkJob = cms.untracked.PSet(
##             limit = cms.untracked.int32(0)
##         ),
##         threshold = cms.untracked.string('DEBUG')
##     ),
##     categories = cms.untracked.vstring('jetMETAnalyzer'),
##     destinations = cms.untracked.vstring('cout')
## )

process.FEVT = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring('keep *_MEtoEDMConverter_*_*'),
    fileName = cms.untracked.string('reco-grumm.root')
)

process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True) ## default is false

)
process.p = cms.Path(process.jetMETAnalyzer)
process.outpath = cms.EndPath(process.FEVT)
process.DQM.collectorHost = ''


