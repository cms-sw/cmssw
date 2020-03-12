# The following comments couldn't be translated into the new config version:

#--------------------------
# DQMServices
#--------------------------

import FWCore.ParameterSet.Config as cms
from RecoLocalTracker.SiPixelRecHits.SiPixelRecHits_cfi import *

process = cms.Process("DQMOnlineSimData")
#-------------------------------------------------
# MAGNETIC FIELD & CO
#-------------------------------------------------
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.connect = "frontier://PromptProd/CMS_COND_21X_GLOBALTAG"
process.GlobalTag.globaltag = "CRAFT_V3P::All"
process.prefer("GlobalTag") 

process.load("Configuration.StandardSequences.MagneticField_38T_cff")
#process.load("Configuration.GlobalRuns.ForceZeroTeslaField_cff")

process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("RecoMuon.DetLayers.muonDetLayerGeometry_cfi")
process.load("Geometry.CommonTopologies.bareGlobalTrackingGeometry_cfi")
process.load("RecoMuon.TrackingTools.MuonServiceProxy_cff")

process.load("RecoLocalTracker.Configuration.RecoLocalTracker_cff")
onlyrechitreco = cms.Sequence( process.siPixelRecHits * process.siStripMatchedRecHits )

#--------------------------
# Tracking Monitor
#--------------------------
process.load("DQM.TrackingMonitor.TrackEfficiencyMonitor_cfi")
process.load("DQM.TrackingMonitor.TrackEfficiencyClient_cfi")

process.MessageLogger = cms.Service("MessageLogger",
    debugModules = cms.untracked.vstring('TrackEffMon'),
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('INFO')
    ),
    destinations = cms.untracked.vstring('cout')
)

process.DQMStore = cms.Service("DQMStore",
    referenceFileName = cms.untracked.string(''),
    verbose = cms.untracked.int32(0)
)

process.outP = cms.OutputModule("AsciiOutputModule")

process.AdaptorConfig = cms.Service("AdaptorConfig")

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('/store/data/Commissioning08/Cosmics/RECO/v1/000/069/276/D6D1312C-6BAA-DD11-972B-000423D94494.root',
'/store/data/Commissioning08/Cosmics/RECO/v1/000/069/276/DEB1E75E-6DAA-DD11-B760-0016177CA778.root',
'/store/data/Commissioning08/Cosmics/RECO/v1/000/069/276/EC28EEF8-6FAA-DD11-A535-001617E30CE8.root',
'/store/data/Commissioning08/Cosmics/RECO/v1/000/069/276/EE51CD95-6EAA-DD11-93BA-001617DBCF1E.root',
'/store/data/Commissioning08/Cosmics/RECO/v1/000/069/276/EE826DE3-6DAA-DD11-AC96-001617C3B778.root',
'/store/data/Commissioning08/Cosmics/RECO/v1/000/069/276/FAA2E035-6DAA-DD11-8CD8-001617E30D38.root',
'/store/data/Commissioning08/Cosmics/RECO/v1/000/069/330/D229C039-B2AA-DD11-B81B-001617C3B65A.root',
'/store/data/Commissioning08/Cosmics/RECO/v1/000/069/332/F240670B-B2AA-DD11-8EED-001617C3B76E.root',
'/store/data/Commissioning08/Cosmics/RECO/v1/000/069/333/44E44B1E-B6AA-DD11-BE1F-00161757BF42.root',
'/store/data/Commissioning08/Cosmics/RECO/v1/000/069/333/54711682-B3AA-DD11-98DA-000423D98DC4.root',
'/store/data/Commissioning08/Cosmics/RECO/v1/000/069/333/5CA882C6-B9AA-DD11-B68A-000423D8F63C.root',
'/store/data/Commissioning08/Cosmics/RECO/v1/000/069/333/985D15CF-B4AA-DD11-B6AA-001617E30F48.root',
'/store/data/Commissioning08/Cosmics/RECO/v1/000/069/333/AA84AE92-B5AA-DD11-9110-000423D999CA.root',
'/store/data/Commissioning08/Cosmics/RECO/v1/000/069/333/B8FED96D-B3AA-DD11-93F9-000423D60FF6.root',
'/store/data/Commissioning08/Cosmics/RECO/v1/000/069/333/BEA373D7-B4AA-DD11-ABB9-000423D94AA8.root',
'/store/data/Commissioning08/Cosmics/RECO/v1/000/069/333/C605EDA1-B6AA-DD11-98A0-000423D99614.root',
'/store/data/Commissioning08/Cosmics/RECO/v1/000/069/335/1C89527D-BAAA-DD11-A7B7-000423D98AF0.root',
'/store/data/Commissioning08/Cosmics/RECO/v1/000/069/335/285C7951-BDAA-DD11-A8D9-000423D98B08.root',
'/store/data/Commissioning08/Cosmics/RECO/v1/000/069/335/2AF47415-B9AA-DD11-9459-000423D6CAF2.root',
'/store/data/Commissioning08/Cosmics/RECO/v1/000/069/335/3A93367B-BAAA-DD11-88EC-000423D991D4.root',
'/store/data/Commissioning08/Cosmics/RECO/v1/000/069/335/4EFE590C-B9AA-DD11-A7A6-000423D99AAA.root',
'/store/data/Commissioning08/Cosmics/RECO/v1/000/069/335/BE8D36E7-BBAA-DD11-91FC-000423D98800.root',
'/store/data/Commissioning08/Cosmics/RECO/v1/000/069/343/10A37DE1-C2AA-DD11-9B0A-000423D944F8.root'
)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(5000)
)
process.p = cms.Path(process.TrackEffMon*process.TrackEffClient)
process.ep = cms.EndPath(process.outP)
process.TrackEffMon.OutputMEsInRootFile = True

