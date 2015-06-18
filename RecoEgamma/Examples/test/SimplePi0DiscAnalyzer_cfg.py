import FWCore.ParameterSet.Config as cms

process = cms.Process("readelectrons")
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.load("Configuration.StandardSequences.MagneticField_38T_cff")
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Geometry.TrackerGeometryBuilder.trackerGeometry_cfi")
process.load("RecoTracker.GeometryESProducer.TrackerRecoGeometryESProducer_cfi")
process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")
process.load("RecoTracker.MeasurementDet.MeasurementTrackerESProducer_cfi")
process.load("SimGeneral.TrackingAnalysis.trackingParticles_cfi")
process.load("SimTracker.TrackAssociatorProducers.trackAssociatorByHits_cfi")
process.load("SimGeneral.MixingModule.mixNoPU_cfi")
process.load("DQMServices.Components.MEtoEDMConverter_cfi")
process.load("RecoEcal.EgammaClusterProducers.preshowerClusterShape_cfi")
process.load("EgammaAnalysis.PhotonIDProducers.piZeroDiscriminators_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(3000)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring (
     '/store/relval/CMSSW_3_3_6/RelValSingleGammaPt35/GEN-SIM-RECO/MC_3XY_V9A-v1/0008/6CF4E2D9-2CE4-DE11-ADEA-001731EF61B4.root'
#     '/store/relval/CMSSW_3_1_4/RelValSingleGammaPt35/GEN-SIM-RECO/MC_31X_V3-v1/0005/6C81A44F-74B0-DE11-818C-001D09F282F5.root'      
    )
)
process.simplePi0DiscAnalyzer = cms.EDAnalyzer("SimplePi0DiscAnalyzer",
    phoProducer = cms.string('photons'),
    photonCollection = cms.string(''),
    outputFile = cms.string('/tmp/akyriaki/SingleGammaPt35_CMSSW_3_3_6_NNoutput.root')
#    outputFile = cms.string('/tmp/akyriaki/SingleGammaPt35_CMSSW_3_1_4_NNoutput.root')
#    outputFile = cms.string('/tmp/akyriaki/Zgamma_Official_Summer09_Photon_NNoutput.root')
)

process.p1 = cms.Path(process.preshowerClusterShape*process.piZeroDiscriminators*process.simplePi0DiscAnalyzer)
process.schedule = cms.Schedule(process.p1)

