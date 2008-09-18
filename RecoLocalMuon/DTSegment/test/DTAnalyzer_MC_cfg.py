import FWCore.ParameterSet.Config as cms

process = cms.Process("CazziMiei")

## General CMS
process.load("Configuration.StandardSequences.FakeConditions_cff")
# process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
# process.GlobalTag.connect = cms.string('frontier://FrontierProd/CMS_COND_21X_DT')
# process.GlobalTag.globaltag = "CRUZET4_V3P::All"
process.load("Configuration.StandardSequences.MagneticField_cff")

# Geometry related
process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cff")
process.load("Geometry.DTGeometry.dtGeometry_cfi")
process.load("Geometry.CSCGeometry.cscGeometry_cfi")
process.load("Geometry.DTGeometryBuilder.idealForDigiDtGeometry_cff")

## DT unpacker
process.load("EventFilter.DTRawToDigi.DTFrontierCabling_cfi")
process.load("EventFilter.DTRawToDigi.dtunpacker_cfi")


process.vdrift = cms.ESSource("PoolDBESSource",
    process.CondDBSetup,
    timetype = cms.string('runnumber'),
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('DTMtimeRcd'),
        tag = cms.string('vDrift_CSA08_S156_mc')
        ),
      ),
##connect = cms.string('frontier://Frontier/CMS_COND_21X_DT'),
    connect = cms.string('frontier://Frontier/CMS_COND_21X_DT'),
    authenticationMethod = cms.untracked.uint32(0)
    )

## DT local Reco
process.load("RecoLocalMuon.Configuration.RecoLocalMuon_cff")
# process.dt2DSegments.debug = cms.untracked.bool(True)
# process.dt2DSegments.Reco2DAlgoConfig.debug = cms.untracked.bool(True)

# process.load("RecoLocalMuon.DTSegment.DTCombinatorialPatternReco2DAlgo_ParamDrift_cfi")
# process.dt1DRecHits.dtDigiLabel = cms.InputTag("simMuonDTDigis")
# process.dt2DSegments.debug = False
# process.dt2DSegments.Reco2DAlgoConfig.debug = False

# process.dt1DClusters = cms.EDFilter("DTClusterer",
#     debug = cms.untracked.bool(False),
#     minLayers = cms.uint32(3),
#     minHits = cms.uint32(3),
#     recHits1DLabel = cms.InputTag("dt1DRecHits")
# )

# process.dt2DExtendedSegments = cms.EDProducer("DTRecSegment2DExtendedProducer",
#     process.DTCombinatorialPatternReco2DAlgo_ParamDrift,
#     debug = cms.untracked.bool(False),
#     recClusLabel = cms.InputTag("dt1DClusters"),
#     recHits1DLabel = cms.InputTag("dt1DRecHits")
# )

# STA reco
process.load("RecoMuon.MuonSeedGenerator.standAloneMuonSeeds_cff")
process.MuonSeed.EnableCSCMeasurement = cms.bool(False)
process.load("RecoMuon.StandAloneMuonProducer.standAloneMuons_cff")
process.standAloneMuons.STATrajBuilderParameters.FilterParameters.EnableCSCMeasurement = cms.bool(False)
process.standAloneMuons.STATrajBuilderParameters.FilterParameters.EnableRPCMeasurement = cms.bool(False)
process.standAloneMuons.STATrajBuilderParameters.BWFilterParameters.EnableCSCMeasurement = cms.bool(False)
process.standAloneMuons.STATrajBuilderParameters.BWFilterParameters.EnableRPCMeasurement = cms.bool(False)

#replace cosmicMuons.TrajectoryBuilderParameters.BackwardMuonTrajectoryUpdatorParameters.Granularity= 2
process.load("TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagator_cfi")

# # ===================================================
# #  7) configuration of your analyzer
# # ===================================================
# process.load("RecoLocalMuon.DTSegment.test.DTFilter_cfi")

# # Hits and Segs ##
# process.load("RecoLocalMuon.DTSegment.test.DTAnalyzerDetailed_cfi")

# Segs ##
process.load("RecoLocalMuon.DTSegment.test.DTSegAnalyzer_cfi")

# # # Clusters ##     
# # process.load("RecoLocalMuon.DTSegment.test.DTClusAnalyzer_cfi")

# # StandAlone ##
# process.load("RecoLocalMuon.DTSegment.test.STAnalyzer_cfi")
# # process.STAnalyzer.debug = cms.untracked.bool(True)

# # Segs Eff ##
# process.load("RecoLocalMuon.DTSegment.test.DTEffAnalyzer_cfi")

process.maxEvents = cms.untracked.PSet(
        input = cms.untracked.int32(-1)
        )

process.options = cms.untracked.PSet(
    #FailPath = cms.untracked.vstring('ProductNotFound'),
    makeTriggerResults = cms.untracked.bool(True),
    wantSummary = cms.untracked.bool(True)
)

# process.MessageLogger = cms.Service("MessageLogger",
#     cout = cms.untracked.PSet(
#         threshold = cms.untracked.string('ERROR')
#     ),
#     destinations = cms.untracked.vstring('cout')
# )

process.source = cms.Source("PoolSource",
fileNames = cms.untracked.vstring( 
        '/store/relval/CMSSW_2_1_8/RelValSingleMuPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V9_t_v3/0000/4E575B6D-5A85-DD11-9566-000423D6B48C.root',
        '/store/relval/CMSSW_2_1_8/RelValSingleMuPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V9_t_v3/0000/9463B994-5A85-DD11-8DC8-000423D6C8EE.root',
        '/store/relval/CMSSW_2_1_8/RelValSingleMuPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V9_t_v3/0000/E2706C9C-5D85-DD11-9F36-000423D6B42C.root')
        # fileNames = cms.untracked.vstring(
    #     '/store/mc/CSA08/Zmumu/GEN-SIM-RECO/CSA08_S156_v1/0002/0A59D9FF-A92B-DD11-8611-001A6434F19C.root',
    #     '/store/mc/CSA08/Zmumu/GEN-SIM-RECO/CSA08_S156_v1/0002/2862D945-AB2B-DD11-832D-001A644EB282.root',
    #     '/store/mc/CSA08/Zmumu/GEN-SIM-RECO/CSA08_S156_v1/0002/56090B0B-B72B-DD11-88ED-001A64894E06.root',
    #     '/store/mc/CSA08/Zmumu/GEN-SIM-RECO/CSA08_S156_v1/0002/AADEE16A-A72B-DD11-839D-00096BB5DCA0.root',
    #     '/store/mc/CSA08/Zmumu/GEN-SIM-RECO/CSA08_S156_v1/0002/E8248443-E42B-DD11-A343-00096BB5B7BA.root',
    #     '/store/mc/CSA08/Zmumu/GEN-SIM-RECO/CSA08_S156_v1/0002/E84DF339-A12B-DD11-9A09-00145EED0A1C.root'
    # )

# fileNames = cms.untracked.vstring( ' file:./output.root')
#fileNames = cms.untracked.vstring( '/store/data/GlobalCruzet1/A/000/000/000/RAW/0000/00217B3F-751B-DD11-A621-001617E30CE8.root')
# fileNames = cms.untracked.vstring(
#     'file:/users/g1cms/lacaprar/800D44DB-2A6C-DD11-9BD6-000423D99EEE.root',
#     'file:/users/g1cms/lacaprar/B65AA3BC-2A6C-DD11-BCF6-001617C3B77C.root',
#     'file:/users/g1cms/lacaprar/E4696611-2B6C-DD11-B826-001617C3B654.root',
#     'file:/users/g1cms/lacaprar/741EBEB6-AC6C-DD11-9762-001617C3B70E.root'
#     )
)

process.muonReco = cms.Sequence(
        process.MuonSeed *
        process.standAloneMuons 
        )

process.muonLocalReco = cms.Sequence(
        process.dtlocalreco_with_2DSegments
        )

process.ana = cms.Sequence(
        process.DTSegAnalyzer
        # process.DTAnalyzerDetailed +
        # process.DTSegAnalyzer +
        # process.DTEffAnalyzer +
        # process.STAnalyzer
        )

process.p = cms.Path(
#        process.DTFilter * (
#            process.muonDTDigis *
            process.muonLocalReco *
            process.muonReco *
            process.ana
            # )
        )

