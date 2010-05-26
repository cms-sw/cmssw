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
process.load("SimTracker.TrackAssociation.TrackAssociatorByHits_cfi")
process.load("SimGeneral.MixingModule.mixNoPU_cfi")
process.load("DQMServices.Components.MEtoEDMConverter_cfi")
process.load("RecoEcal.EgammaClusterProducers.preshowerClusterShape_cfi")
process.load("EgammaAnalysis.PhotonIDProducers.piZeroDiscriminators_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(3000)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring (
     '/store/relval/CMSSW_3_1_0_pre7/RelValSingleGammaPt35/GEN-SIM-RECO/IDEAL_31X_v1/0004/DE2A9491-CD41-DE11-978D-001D09F2525D.root'      
    )
)
process.simplePi0DiscAnalyzer = cms.EDAnalyzer("SimplePi0DiscAnalyzer",
    phoProducer = cms.string('photons'),
    photonCollection = cms.string(''),
    outputFile = cms.string('SingleGammaPt35_CMSSW_3_1_0_pre7_NNoutput.root')
)

process.p1 = cms.Path(process.preshowerClusterShape*process.piZeroDiscriminators*process.simplePi0DiscAnalyzer)
process.schedule = cms.Schedule(process.p1)

