import FWCore.ParameterSet.Config as cms

process = cms.Process("Demo")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi");
process.load("Geometry.CaloEventSetup.CaloGeometry_cfi");
process.load("Geometry.CaloEventSetup.CaloTopology_cfi");
process.load("TrackingTools/TransientTrack/TransientTrackBuilder_cfi")
process.load("Configuration.Geometry.GeometryIdeal_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        '/store/mc/Run3Winter21DRMiniAOD/HTo2LongLivedTo4b_MH-250_MFF-60_CTau-1000mm_TuneCP5_14TeV-pythia8/GEN-SIM-RECO/FlatPU30to80FEVT_112X_mcRun3_2021_realistic_v16-v2/130000/058470ca-8aaa-4727-80d8-f42621bafd39.root'
    )
)
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag.globaltag = '121X_mcRun3_2021_realistic_v1'

process.hltTimingProducer = cms.EDProducer('HLTCaloJetTimingProducer',
        jets = cms.InputTag( "ak4CaloJets" ),
        ebRecHitsColl = cms.InputTag( 'ecalRecHit','EcalRecHitsEB' ),
        eeRecHitsColl = cms.InputTag( 'ecalRecHit','EcalRecHitsEE' ),
        barrelJets = cms.bool(True),
        endcapJets = cms.bool(False),
        ecalCellEnergyThresh =cms.double(0.5),
        ecalCellTimeThresh = cms.double(12.5),
        ecalCellTimeErrorThresh = cms.double(100.),
        matchingRadius = cms.double(0.4),
)

process.hltTimingFilter = cms.EDFilter('HLTCaloJetTimingFilter',
    saveTags = cms.bool( True ),
    minJets = cms.uint32(1),
    minJetPt = cms.double(40.0),
    jetTimeThresh = cms.double(1.),
    jetCellsForTimingThresh = cms.uint32(5),
    jetEcalEtForTimingThresh = cms.double(10.),
    jets = cms.InputTag( "ak4CaloJets" ),
    jetTimes = cms.InputTag( "hltTimingProducer" ),
    jetEcalEtForTiming = cms.InputTag( "hltTimingProducer" ,"jetEcalEtForTiming"),
    jetCellsForTiming = cms.InputTag( "hltTimingProducer" ,"jetCellsForTiming"),
)
process.output = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "timingOutput.root" ),
)

process.p = cms.Path(process.hltTimingProducer+process.hltTimingFilter)
process.Output = cms.EndPath(process.output)
