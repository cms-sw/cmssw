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

process.load("CalibMuon.Configuration.DT_FakeConditions_cff")

## DT local Reco
# process.load("RecoLocalMuon.Configuration.RecoLocalMuon_cff")
process.load("RecoLocalMuon.DTRecHit.dt1DRecHits_ParamDrift_cfi")
#process.dt1DRecHits.recAlgoConfig.tTrigModeConfig.kFactor = cms.double(-2.5)
process.load("RecoLocalMuon.DTSegment.dt2DSegments_CombPatternReco2D_ParamDrift_cfi")
process.load("RecoLocalMuon.DTSegment.dt4DSegments_CombPatternReco4D_ParamDrift_cfi")
process.dt4DSegments.recHits2DLabel= cms.InputTag("dt2DExtendedSegments")

# # only for debuging purpose and for specific studies
# process.dtlocalreco = cms.Sequence(
#     process.dt1DRecHits
#     * process.dt4DSegments
#     )

process.dt1DClusters = cms.EDFilter("DTClusterer",
    debug = cms.untracked.bool(False),
    minLayers = cms.uint32(3),
    minHits = cms.uint32(3),
    recHits1DLabel = cms.InputTag("dt1DRecHits")
)

from RecoLocalMuon.DTSegment.DTCombinatorialPatternReco2DAlgo_LinearDriftFromDB_cfi import *
process.dt2DExtendedSegments = cms.EDProducer("DTRecSegment2DExtendedProducer",
    DTCombinatorialPatternReco2DAlgo_LinearDriftFromDB,
    #process.DTCombinatorialPatternReco2DAlgo_LinearDriftFromDB,
    debug = cms.untracked.bool(False),
    recClusLabel = cms.InputTag("dt1DClusters"),
    recHits1DLabel = cms.InputTag("dt1DRecHits")
)

# DT sequence with the 2D segment reconstruction
process.dtlocalreco_with_2DExtendedSegments = cms.Sequence(
    process.dt1DRecHits
    * process.dt1DClusters
    * process.dt2DExtendedSegments
    * process.dt4DSegments
    )

# # DT sequence with the 2D segment reconstruction
# process.dtlocalreco_with_2DSegments = cms.Sequence(
#     process.dt1DRecHits
#     * process.dt2DSegments
#     * process.dt4DSegments
#     )

# STA reco
process.load("RecoMuon.MuonSeedGenerator.standAloneMuonSeeds_cff")
process.MuonSeed.EnableCSCMeasurement = cms.bool(False)
process.load("RecoMuon.StandAloneMuonProducer.standAloneMuons_cff")
process.standAloneMuons.STATrajBuilderParameters.FilterParameters.EnableCSCMeasurement = cms.bool(False)
process.standAloneMuons.STATrajBuilderParameters.FilterParameters.EnableRPCMeasurement = cms.bool(False)
process.standAloneMuons.STATrajBuilderParameters.BWFilterParameters.EnableCSCMeasurement = cms.bool(False)
process.standAloneMuons.STATrajBuilderParameters.BWFilterParameters.EnableRPCMeasurement = cms.bool(False)

#replace cosmicMuons.TrajectoryBuilderParameters.BackwardMuonTrajectoryUpdatorParameters.Granularity= 2
#process.load("TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagator_cfi")

# ===================================================
#  7) configuration of your analyzer
# ===================================================
process.load("RecoLocalMuon.DTSegment.DTFilter_cfi")

# Hits and Segs ##
process.load("RecoLocalMuon.DTSegment.DTAnalyzerDetailed_cfi")
process.DTAnalyzerDetailed.recHits2DLabel = cms.string('dt2DExtendedSegments')

# Segs ##
process.load("RecoLocalMuon.DTSegment.DTSegAnalyzer_cfi")
process.DTSegAnalyzer.recHits2DLabel = cms.string('dt2DExtendedSegments')

# Clusters ##     
process.load("RecoLocalMuon.DTSegment.DTClusAnalyzer_cfi")
process.DTClusAnalyzer.recHits2DLabel = cms.string('dt2DExtendedSegments')

# StandAlone ##
process.load("RecoLocalMuon.DTSegment.STAnalyzer_cfi")
process.STAnalyzer.SALabel = cms.string('standAloneMuons')
process.STAnalyzer.recHits2DLabel = cms.string('dt2DExtendedSegments')
# # process.STAnalyzer.debug = cms.untracked.bool(True)

# Segs Eff ##
process.load("RecoLocalMuon.DTSegment.DTEffAnalyzer_cfi")

# Validation RecHits
process.load("Validation.DTRecHits.test.DTRecHitQuality_cfi")
process.rechivalidation.segment2DLabel = cms.untracked.string('dt2DExtendedSegments')
process.seg2dvalidation.segment2DLabel = cms.untracked.string('dt2DExtendedSegments')
#process.seg2dvalidation.debug = cms.untracked.bool(True)

process.maxEvents = cms.untracked.PSet(
        input = cms.untracked.int32(10000)
        )

process.options = cms.untracked.PSet(
    #FailPath = cms.untracked.vstring('ProductNotFound'),
    makeTriggerResults = cms.untracked.bool(True),
    wantSummary = cms.untracked.bool(True)
)

process.MessageLogger = cms.Service("MessageLogger",
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('WARNING')
    ),
    destinations = cms.untracked.vstring('cout')
)

process.source = cms.Source("PoolSource",
fileNames = cms.untracked.vstring( 
### SingleMuon Pt1000
# '/store/relval/CMSSW_2_1_9/RelValSingleMuPt1000/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V9_v2/0000/162D72AC-B185-DD11-9D6E-000423D6AF24.root',
# '/store/relval/CMSSW_2_1_9/RelValSingleMuPt1000/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V9_v2/0000/46E3E527-AF85-DD11-97BC-000423D98EC4.root',
# '/store/relval/CMSSW_2_1_9/RelValSingleMuPt1000/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V9_v2/0000/C6D7FAEB-AC85-DD11-8F06-001617E30D40.root',
# '/store/relval/CMSSW_2_1_9/RelValSingleMuPt1000/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V9_v2/0001/4AC212E4-0487-DD11-9A6D-000423D94A04.root'

### SingleMuon Pt100
     # '/store/relval/CMSSW_2_1_9/RelValSingleMuPt100/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V9_v2/0000/BA6DA407-A985-DD11-A0D8-000423D9A2AE.root',
     #   '/store/relval/CMSSW_2_1_9/RelValSingleMuPt100/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V9_v2/0000/CAADA54A-AB85-DD11-95D6-000423D98F98.root',
     #   '/store/relval/CMSSW_2_1_9/RelValSingleMuPt100/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V9_v2/0001/52BAB7DF-0487-DD11-95A3-000423D9989E.root'



### SingleMuon Pt10
        '/store/relval/CMSSW_2_1_8/RelValSingleMuPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V9_t_v3/0000/4E575B6D-5A85-DD11-9566-000423D6B48C.root',
        '/store/relval/CMSSW_2_1_8/RelValSingleMuPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V9_t_v3/0000/9463B994-5A85-DD11-8DC8-000423D6C8EE.root',
        '/store/relval/CMSSW_2_1_8/RelValSingleMuPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V9_t_v3/0000/E2706C9C-5D85-DD11-9F36-000423D6B42C.root'
        )

)

process.muonReco = cms.Sequence(
        process.MuonSeed *
        process.standAloneMuons 
        )

process.muonLocalReco = cms.Sequence(
        # process.dtlocalreco_with_2DSegments
        process.dtlocalreco_with_2DExtendedSegments
        )

process.ana = cms.Sequence(
        process.DTAnalyzerDetailed
        + process.DTSegAnalyzer
        + process.DTClusAnalyzer
        + process.DTEffAnalyzer
        + process.dtLocalRecoValidation
        #+ process.STAnalyzer
        # process.DTSegAnalyzer +
        # process.DTEffAnalyzer +
        )

process.p = cms.Path(
#        process.DTFilter * (
#            process.muonDTDigis *
            process.muonLocalReco *
         #   process.muonReco *
            process.ana
            # )
        )

