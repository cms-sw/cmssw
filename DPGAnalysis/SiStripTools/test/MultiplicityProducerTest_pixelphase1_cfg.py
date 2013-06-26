import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing

process = cms.Process("MultiplicityProducerTest")

#prepare options

options = VarParsing.VarParsing("analysis")

options.register ('globalTag',
                  "DONOTEXIST::All",
                  VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                  VarParsing.VarParsing.varType.string,          # string, int, or float
                  "GlobalTag")
options.register ('fromRAW',
                  "0",
                  VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                  VarParsing.VarParsing.varType.int,          # string, int, or float
                  "=1 if from RAW")
options.register ('testTag',
                  "0",
                  VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                  VarParsing.VarParsing.varType.int,          # string, int, or float
                  "=1 if test tag to be used")

options.parseArguments()

#

process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True),
    fileMode = cms.untracked.string("FULLMERGE")
    )

process.load("FWCore.MessageService.MessageLogger_cfi")

process.MessageLogger.cout.placeholder = cms.untracked.bool(False)
process.MessageLogger.cout.threshold = cms.untracked.string("INFO")
process.MessageLogger.cout.default = cms.untracked.PSet(
    limit = cms.untracked.int32(10000000)
    )
process.MessageLogger.cout.FwkReport = cms.untracked.PSet(
    reportEvery = cms.untracked.int32(10000)
    )

process.MessageLogger.cerr.placeholder = cms.untracked.bool(False)
process.MessageLogger.cerr.threshold = cms.untracked.string("WARNING")
process.MessageLogger.cerr.default = cms.untracked.PSet(
    limit = cms.untracked.int32(10000000)
    )
process.MessageLogger.cerr.FwkReport = cms.untracked.PSet(
    reportEvery = cms.untracked.int32(100000)
    )

process.MessageLogger.debugModules = cms.untracked.vstring("clustsummmultprod","spclustermultprod","ssclustermultprod")

#----Remove too verbose PrimaryVertexProducer

process.MessageLogger.suppressInfo.append("pixelVerticesAdaptive")
process.MessageLogger.suppressInfo.append("pixelVerticesAdaptiveNoBS")

#----Remove too verbose BeamSpotOnlineProducer

process.MessageLogger.suppressInfo.append("testBeamSpot")
process.MessageLogger.suppressInfo.append("onlineBeamSpot")
process.MessageLogger.suppressWarning.append("testBeamSpot")
process.MessageLogger.suppressWarning.append("onlineBeamSpot")

#----Remove too verbose TrackRefitter

process.MessageLogger.suppressInfo.append("newTracksFromV0")
process.MessageLogger.suppressInfo.append("newTracksFromOtobV0")

#------------------------------------------------------------------

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(options.maxEvents) )

process.source = cms.Source("PoolSource",
                    fileNames = cms.untracked.vstring(options.inputFiles),
#                    skipBadFiles = cms.untracked.bool(True),
                    inputCommands = cms.untracked.vstring("keep *", "drop *_MEtoEDMConverter_*_*")
                    )

#--------------------------------------
process.seqRECO = cms.Sequence()

if options.fromRAW == 1:
    process.load("Configuration.StandardSequences.RawToDigi_Data_cff")
    process.load("Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff")
    process.load("Configuration.StandardSequences.GeometryDB_cff")
    process.load("Configuration.StandardSequences.Reconstruction_cff")
    process.load("Configuration.StandardSequences.L1Reco_cff")
    process.seqRECO = cms.Sequence(process.gtEvmDigis + process.L1Reco
                                   + process.siStripDigis + process.siStripZeroSuppression + process.siStripClusters
                                   + process.siPixelDigis + process.siPixelClusters )


#

process.load("DPGAnalysis.SiStripTools.sipixelclustermultiplicityprod_cfi")
process.load("DPGAnalysis.SiStripTools.sistripclustermultiplicityprod_cfi")

#process.spclustermultprod.withClusterSize=cms.untracked.bool(True)
#process.ssclustermultprod.withClusterSize=cms.untracked.bool(True)

process.spclustermultprodnew = process.spclustermultprod.clone(wantedSubDets = cms.VPSet(    
    cms.PSet(detSelection = cms.uint32(101),detLabel = cms.string("BPIX"),selection=cms.untracked.vstring("0x1e000000-0x12000000")),
    cms.PSet(detSelection = cms.uint32(102),detLabel = cms.string("FPIX"),selection=cms.untracked.vstring("0x1e000000-0x14000000")),
    cms.PSet(detSelection = cms.uint32(11),detLabel = cms.string("BPIXL1"),selection=cms.untracked.vstring("0x1e0f0000-0x12010000")),
    cms.PSet(detSelection = cms.uint32(12),detLabel = cms.string("BPIXL2"),selection=cms.untracked.vstring("0x1e0f0000-0x12020000")),
    cms.PSet(detSelection = cms.uint32(13),detLabel = cms.string("BPIXL3"),selection=cms.untracked.vstring("0x1e0f0000-0x12030000")),
    cms.PSet(detSelection = cms.uint32(21),detLabel = cms.string("FPIXm"),selection=cms.untracked.vstring("0x1f800000-0x14800000")),
    cms.PSet(detSelection = cms.uint32(22),detLabel = cms.string("FPIXp"),selection=cms.untracked.vstring("0x1f800000-0x15000000")),
    cms.PSet(detSelection = cms.uint32(211),detLabel = cms.string("FPIXmD1"),selection=cms.untracked.vstring("0x1f8f0000-0x14810000")),
    cms.PSet(detSelection = cms.uint32(212),detLabel = cms.string("FPIXmD2"),selection=cms.untracked.vstring("0x1f8f0000-0x14820000")),
    cms.PSet(detSelection = cms.uint32(213),detLabel = cms.string("FPIXmD3"),selection=cms.untracked.vstring("0x1f8f0000-0x14830000")),
    cms.PSet(detSelection = cms.uint32(221),detLabel = cms.string("FPIXpD1"),selection=cms.untracked.vstring("0x1f8f0000-0x15010000")),
    cms.PSet(detSelection = cms.uint32(222),detLabel = cms.string("FPIXpD2"),selection=cms.untracked.vstring("0x1f8f0000-0x15020000")),
    cms.PSet(detSelection = cms.uint32(223),detLabel = cms.string("FPIXpD3"),selection=cms.untracked.vstring("0x1f8f0000-0x15030000")),
    cms.PSet(detSelection = cms.uint32(2111),detLabel = cms.string("FPIXmD1pan1"),selection=cms.untracked.vstring("0x1f8f0300-0x14810100")),
    cms.PSet(detSelection = cms.uint32(2121),detLabel = cms.string("FPIXmD2pan1"),selection=cms.untracked.vstring("0x1f8f0300-0x14820100")),
    cms.PSet(detSelection = cms.uint32(2131),detLabel = cms.string("FPIXmD3pan1"),selection=cms.untracked.vstring("0x1f8f0300-0x14830100")),
    cms.PSet(detSelection = cms.uint32(2211),detLabel = cms.string("FPIXpD1pan1"),selection=cms.untracked.vstring("0x1f8f0300-0x15010100")),
    cms.PSet(detSelection = cms.uint32(2221),detLabel = cms.string("FPIXpD2pan1"),selection=cms.untracked.vstring("0x1f8f0300-0x15020100")),
    cms.PSet(detSelection = cms.uint32(2231),detLabel = cms.string("FPIXpD3pan1"),selection=cms.untracked.vstring("0x1f8f0300-0x15030100")),
    cms.PSet(detSelection = cms.uint32(2112),detLabel = cms.string("FPIXmD1pan2"),selection=cms.untracked.vstring("0x1f8f0300-0x14810200")),
    cms.PSet(detSelection = cms.uint32(2122),detLabel = cms.string("FPIXmD2pan2"),selection=cms.untracked.vstring("0x1f8f0300-0x14820200")),
    cms.PSet(detSelection = cms.uint32(2132),detLabel = cms.string("FPIXmD3pan2"),selection=cms.untracked.vstring("0x1f8f0300-0x14830200")),
    cms.PSet(detSelection = cms.uint32(2212),detLabel = cms.string("FPIXpD1pan2"),selection=cms.untracked.vstring("0x1f8f0300-0x15010200")),
    cms.PSet(detSelection = cms.uint32(2222),detLabel = cms.string("FPIXpD2pan2"),selection=cms.untracked.vstring("0x1f8f0300-0x15020200")),
    cms.PSet(detSelection = cms.uint32(2232),detLabel = cms.string("FPIXpD3pan2"),selection=cms.untracked.vstring("0x1f8f0300-0x15030200"))
    )
)
process.ssclustermultprodnew = process.ssclustermultprod.clone(wantedSubDets = cms.VPSet(    
    cms.PSet(detSelection = cms.uint32(103),detLabel = cms.string("TIB"),selection=cms.untracked.vstring("0x1e000000-0x16000000")),
    cms.PSet(detSelection = cms.uint32(114),detLabel = cms.string("TIDm"),selection=cms.untracked.vstring("0x1e006000-0x18002000")),
    cms.PSet(detSelection = cms.uint32(124),detLabel = cms.string("TIDp"),selection=cms.untracked.vstring("0x1e006000-0x18004000")),
    cms.PSet(detSelection = cms.uint32(116),detLabel = cms.string("TECm"),selection=cms.untracked.vstring("0x1e0c0000-0x1c040000")),
    cms.PSet(detSelection = cms.uint32(126),detLabel = cms.string("TECp"),selection=cms.untracked.vstring("0x1e0c0000-0x1c080000")),
    cms.PSet(detSelection = cms.uint32(999),detLabel = cms.string("Module"),selection=cms.untracked.vstring("0x1fffffff-0x1600546d"))
    )
)

process.load("DPGAnalysis.SiStripTools.clustersummarymultiplicityprod_cfi")

process.clustsummmultprod.wantedSubDets = cms.VPSet(
    cms.PSet(detSelection = cms.uint32(0),detLabel = cms.string("TK"), subDetEnum = cms.int32(0), subDetVariable = cms.string("cHits")),
    cms.PSet(detSelection = cms.uint32(1),detLabel = cms.string("TIB"), subDetEnum = cms.int32(1), subDetVariable = cms.string("cHits")),
    cms.PSet(detSelection = cms.uint32(2),detLabel = cms.string("TOB"), subDetEnum = cms.int32(2), subDetVariable = cms.string("cHits")),
    cms.PSet(detSelection = cms.uint32(3),detLabel = cms.string("TID"), subDetEnum = cms.int32(3), subDetVariable = cms.string("cHits")),
    cms.PSet(detSelection = cms.uint32(4),detLabel = cms.string("TEC"), subDetEnum = cms.int32(4), subDetVariable = cms.string("cHits")),
    cms.PSet(detSelection = cms.uint32(1005),detLabel = cms.string("Pixel"), subDetEnum = cms.int32(5), subDetVariable = cms.string("pHits")),
    cms.PSet(detSelection = cms.uint32(1006),detLabel = cms.string("FPIX"), subDetEnum = cms.int32(6), subDetVariable = cms.string("pHits")),
    cms.PSet(detSelection = cms.uint32(1007),detLabel = cms.string("BPIX"), subDetEnum = cms.int32(7), subDetVariable = cms.string("pHits"))
    )

#process.ssclustermultprod.clusterdigiCollection = cms.InputTag("calZeroBiasClusters")
#process.ssclustermultprodnew.clusterdigiCollection = cms.InputTag("calZeroBiasClusters")

process.spclustermultprodontrack = process.spclustermultprod.clone(clusterdigiCollection = cms.InputTag("AlignmentTrackSelector"))
process.ssclustermultprodontrack = process.ssclustermultprod.clone(clusterdigiCollection = cms.InputTag("AlignmentTrackSelector"))
process.spclustermultprodnewontrack = process.spclustermultprodnew.clone(clusterdigiCollection = cms.InputTag("AlignmentTrackSelector"))
process.ssclustermultprodnewontrack = process.ssclustermultprodnew.clone(clusterdigiCollection = cms.InputTag("AlignmentTrackSelector"))

process.seqMultProd = cms.Sequence(process.spclustermultprod+process.ssclustermultprod
                                   +process.spclustermultprodontrack+process.ssclustermultprodontrack
                                   +process.spclustermultprodnew+process.ssclustermultprodnew
                                   +process.spclustermultprodnewontrack+process.ssclustermultprodnewontrack
                                   +process.clustsummmultprod)
#process.seqMultProd = cms.Sequence(process.ssclustermultprod+process.ssclustermultprodnew)
#process.seqMultProd = cms.Sequence(process.clustsummmultprod)

process.goodVertices = cms.EDFilter("VertexSelector",
   src = cms.InputTag("offlinePrimaryVertices"),
   cut = cms.string("!isFake && ndof > 4 && abs(z) <= 24 && position.Rho <= 2"),  
   filter = cms.bool(False),   # otherwise it won't filter the events, just produce an empty vertex collection.
)

process.load("RecoLocalTracker.SiStripClusterizer.SiStripClusterToDigiProducer_cfi")

# Track selector from alignment group to get on track clusters

process.load("Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi")


process.seqProducers = cms.Sequence(process.AlignmentTrackSelector
                                    + process.seqRECO + process.seqMultProd + process.siStripClustersToDigis
                                    + process.goodVertices
                                    )

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import *
process.hltSelection = triggerResultsFilter.clone(
                                          triggerConditions = cms.vstring("HLT_ZeroBias_*"),
                                          hltResults = cms.InputTag( "TriggerResults", "", "HLT" ),
                                          l1tResults = cms.InputTag( "" ),
                                          throw = cms.bool(False)
                                          )

process.manystripclus = cms.EDFilter('BySiPixelClusterVsSiStripClusterMultiplicityEventFilter',
                                     multiplicityConfig = cms.PSet(
    firstMultiplicityConfig = cms.PSet(
    collectionName = cms.InputTag("siPixelClusters"),
    moduleThreshold = cms.untracked.int32(-1),
    useQuality = cms.untracked.bool(False),
    qualityLabel = cms.untracked.string("")
    ),
    secondMultiplicityConfig = cms.PSet(
    collectionName = cms.InputTag("siStripClusters"),
    moduleThreshold = cms.untracked.int32(-1),
    useQuality = cms.untracked.bool(False),
    qualityLabel = cms.untracked.string("")
    )
    ),
                                     cut = cms.string("( mult2 > 20000+7*mult1)")
                                     )

process.manystripclus53X = cms.EDFilter('ByClusterSummaryMultiplicityPairEventFilter',
                                        multiplicityConfig = cms.PSet(
    firstMultiplicityConfig = cms.PSet(
    clusterSummaryCollection = cms.InputTag("clusterSummaryProducer"),
    subDetEnum = cms.int32(5),
    subDetVariable = cms.string("pHits")
    ),
    secondMultiplicityConfig = cms.PSet(
    clusterSummaryCollection = cms.InputTag("clusterSummaryProducer"),
    subDetEnum = cms.int32(0),
    subDetVariable = cms.string("cHits")
    ),
    ),
                                        cut = cms.string("( mult2 > 20000+7*mult1)")
                                        )

process.load("DPGAnalysis.SiStripTools.ssclusmultinvestigator_cfi")
#process.ssclusmultinvestigator.vertexCollection = cms.InputTag("goodVertices")
#process.ssclusmultinvestigator.wantInvestHist = cms.bool(True)
#process.ssclusmultinvestigator.wantVtxCorrHist = cms.bool(True)
#process.ssclusmultinvestigator.digiVtxCorrConfig = cms.PSet(
#    wantedSubDets = process.ssclusmultinvestigator.wantedSubDets,
#    hitName = cms.untracked.string("cluster"),
#    numberOfBins = cms.untracked.int32(100),   
#    scaleFactor = cms.untracked.int32(1),
#    maxNvtx = cms.untracked.int32(100)
#    )
#process.ssclusmultinvestigator.wantLumiCorrHist = cms.bool(True)
#process.ssclusmultinvestigator.digiLumiCorrConfig = cms.PSet(
#    lumiProducer = cms.InputTag("lumiProducer"),
#    wantedSubDets = process.ssclusmultinvestigator.wantedSubDets,
#    hitName = cms.untracked.string("cluster"),
#    numberOfBins = cms.untracked.int32(100),   
#    scaleFactor = cms.untracked.int32(1),
#    maxLumi = cms.untracked.double(25)
#    )


#process.ssclusmultinvestigator.runHisto = cms.untracked.bool(True)
process.ssclusmultinvestigator.scaleFactor=cms.untracked.int32(1)

process.ssclusmultinvestigatornew = process.ssclusmultinvestigator.clone()
process.ssclusmultinvestigatornew.runHisto = cms.untracked.bool(True)
process.ssclusmultinvestigatornew.wantedSubDets = cms.untracked.VPSet(    
    cms.PSet(detSelection = cms.uint32(103),detLabel = cms.string("TIB"), binMax = cms.int32(1787904/32)),
    cms.PSet(detSelection = cms.uint32(114),detLabel = cms.string("TIDm"), binMax = cms.int32(565248/32)),
    cms.PSet(detSelection = cms.uint32(124),detLabel = cms.string("TIDp"), binMax = cms.int32(565248/32)),
    cms.PSet(detSelection = cms.uint32(116),detLabel = cms.string("TECm"), binMax = cms.int32(3866624/64)),
    cms.PSet(detSelection = cms.uint32(126),detLabel = cms.string("TECp"), binMax = cms.int32(3866624/64)),
    cms.PSet(detSelection = cms.uint32(999),detLabel = cms.string("Module"), binMax = cms.int32(200))
    )
#process.ssclusmultinvestigatornew.digiVtxCorrConfig.wantedSubDets = cms.untracked.VPSet(    
#    cms.PSet(detSelection = cms.uint32(103),detLabel = cms.string("TIB"), binMax = cms.int32(1787904/32)),
#    cms.PSet(detSelection = cms.uint32(114),detLabel = cms.string("TIDm"), binMax = cms.int32(565248/32)),
#    cms.PSet(detSelection = cms.uint32(124),detLabel = cms.string("TIDp"), binMax = cms.int32(565248/32)),
#    cms.PSet(detSelection = cms.uint32(116),detLabel = cms.string("TECm"), binMax = cms.int32(3866624/64)),
#    cms.PSet(detSelection = cms.uint32(126),detLabel = cms.string("TECp"), binMax = cms.int32(3866624/64)),
#    cms.PSet(detSelection = cms.uint32(999),detLabel = cms.string("Module"), binMax = cms.int32(200))
#    )
#process.ssclusmultinvestigatornew.digiLumiCorrConfig.wantedSubDets = cms.untracked.VPSet(    
#    cms.PSet(detSelection = cms.uint32(103),detLabel = cms.string("TIB"), binMax = cms.int32(1787904/32)),
#    cms.PSet(detSelection = cms.uint32(114),detLabel = cms.string("TIDm"), binMax = cms.int32(565248/32)),
#    cms.PSet(detSelection = cms.uint32(124),detLabel = cms.string("TIDp"), binMax = cms.int32(565248/32)),
#    cms.PSet(detSelection = cms.uint32(116),detLabel = cms.string("TECm"), binMax = cms.int32(3866624/64)),
#    cms.PSet(detSelection = cms.uint32(126),detLabel = cms.string("TECp"), binMax = cms.int32(3866624/64)),
#    cms.PSet(detSelection = cms.uint32(999),detLabel = cms.string("Module"), binMax = cms.int32(200))
#    )
process.ssclusmultinvestigatornew.multiplicityMap = cms.InputTag("ssclustermultprodnew")


process.load("DPGAnalysis.SiStripTools.spclusmultinvestigator_cfi")
process.spclusmultinvestigator.vertexCollection = cms.InputTag("goodVertices")
process.spclusmultinvestigator.wantInvestHist = cms.bool(True)
process.spclusmultinvestigator.wantVtxCorrHist = cms.bool(True)
process.spclusmultinvestigator.digiVtxCorrConfig = cms.PSet(
    wantedSubDets = process.spclusmultinvestigator.wantedSubDets,
    hitName = cms.untracked.string("cluster"),
    numberOfBins = cms.untracked.int32(100),   
    scaleFactor = cms.untracked.int32(5)
    )
process.spclusmultinvestigator.wantLumiCorrHist = cms.bool(True)
process.spclusmultinvestigator.digiLumiCorrConfig = cms.PSet(
    lumiProducer = cms.InputTag("lumiProducer"),
    wantedSubDets = process.spclusmultinvestigator.wantedSubDets,
    hitName = cms.untracked.string("cluster"),
    numberOfBins = cms.untracked.int32(100),   
    scaleFactor = cms.untracked.int32(5)
    )
#process.spclusmultinvestigator.runHisto = cms.untracked.bool(True)
process.spclusmultinvestigator.scaleFactor=cms.untracked.int32(5)

process.spclusmultinvestigatornew = process.spclusmultinvestigator.clone()
process.spclusmultinvestigatornew.wantedSubDets = cms.untracked.VPSet(    
    cms.PSet(detSelection = cms.uint32(101),detLabel = cms.string("BPIX"), binMax = cms.int32(100000)),
    cms.PSet(detSelection = cms.uint32(11),detLabel = cms.string("BPIXL1"), binMax = cms.int32(100000)),
    cms.PSet(detSelection = cms.uint32(12),detLabel = cms.string("BPIXL2"), binMax = cms.int32(100000)),
    cms.PSet(detSelection = cms.uint32(13),detLabel = cms.string("BPIXL3"), binMax = cms.int32(100000)),
    cms.PSet(detSelection = cms.uint32(102),detLabel = cms.string("FPIX"), binMax = cms.int32(100000)),
    cms.PSet(detSelection = cms.uint32(21),detLabel = cms.string("FPIXm"), binMax = cms.int32(100000)),
    cms.PSet(detSelection = cms.uint32(22),detLabel = cms.string("FPIXp"), binMax = cms.int32(100000)),
    cms.PSet(detSelection = cms.uint32(211),detLabel = cms.string("FPIXmD1"), binMax = cms.int32(100000)),
    cms.PSet(detSelection = cms.uint32(212),detLabel = cms.string("FPIXmD2"), binMax = cms.int32(100000)),
    cms.PSet(detSelection = cms.uint32(213),detLabel = cms.string("FPIXmD3"), binMax = cms.int32(100000)),
    cms.PSet(detSelection = cms.uint32(221),detLabel = cms.string("FPIXpD1"), binMax = cms.int32(100000)),
    cms.PSet(detSelection = cms.uint32(222),detLabel = cms.string("FPIXpD2"), binMax = cms.int32(100000)),
    cms.PSet(detSelection = cms.uint32(223),detLabel = cms.string("FPIXpD3"), binMax = cms.int32(100000)),
    cms.PSet(detSelection = cms.uint32(2111),detLabel = cms.string("FPIXmD1pan1"), binMax = cms.int32(100000)),
    cms.PSet(detSelection = cms.uint32(2121),detLabel = cms.string("FPIXmD2pan1"), binMax = cms.int32(100000)),
    cms.PSet(detSelection = cms.uint32(2131),detLabel = cms.string("FPIXmD3pan1"), binMax = cms.int32(100000)),
    cms.PSet(detSelection = cms.uint32(2211),detLabel = cms.string("FPIXpD1pan1"), binMax = cms.int32(100000)),
    cms.PSet(detSelection = cms.uint32(2221),detLabel = cms.string("FPIXpD2pan1"), binMax = cms.int32(100000)),
    cms.PSet(detSelection = cms.uint32(2231),detLabel = cms.string("FPIXpD3pan1"), binMax = cms.int32(100000)),
    cms.PSet(detSelection = cms.uint32(2112),detLabel = cms.string("FPIXmD1pan2"), binMax = cms.int32(100000)),
    cms.PSet(detSelection = cms.uint32(2122),detLabel = cms.string("FPIXmD2pan2"), binMax = cms.int32(100000)),
    cms.PSet(detSelection = cms.uint32(2132),detLabel = cms.string("FPIXmD3pan2"), binMax = cms.int32(100000)),
    cms.PSet(detSelection = cms.uint32(2212),detLabel = cms.string("FPIXpD1pan2"), binMax = cms.int32(100000)),
    cms.PSet(detSelection = cms.uint32(2222),detLabel = cms.string("FPIXpD2pan2"), binMax = cms.int32(100000)),
    cms.PSet(detSelection = cms.uint32(2232),detLabel = cms.string("FPIXpD3pan2"), binMax = cms.int32(100000))
    )
process.spclusmultinvestigatornew.digiVtxCorrConfig.wantedSubDets = cms.untracked.VPSet(    
    cms.PSet(detSelection = cms.uint32(101),detLabel = cms.string("BPIX"), binMax = cms.int32(100000)),
    cms.PSet(detSelection = cms.uint32(11),detLabel = cms.string("BPIXL1"), binMax = cms.int32(100000)),
    cms.PSet(detSelection = cms.uint32(12),detLabel = cms.string("BPIXL2"), binMax = cms.int32(100000)),
    cms.PSet(detSelection = cms.uint32(13),detLabel = cms.string("BPIXL3"), binMax = cms.int32(100000)),
    cms.PSet(detSelection = cms.uint32(102),detLabel = cms.string("FPIX"), binMax = cms.int32(100000)),
    cms.PSet(detSelection = cms.uint32(21),detLabel = cms.string("FPIXm"), binMax = cms.int32(100000)),
    cms.PSet(detSelection = cms.uint32(22),detLabel = cms.string("FPIXp"), binMax = cms.int32(100000)),
    cms.PSet(detSelection = cms.uint32(211),detLabel = cms.string("FPIXmD1"), binMax = cms.int32(100000)),
    cms.PSet(detSelection = cms.uint32(212),detLabel = cms.string("FPIXmD2"), binMax = cms.int32(100000)),
    cms.PSet(detSelection = cms.uint32(213),detLabel = cms.string("FPIXmD3"), binMax = cms.int32(100000)),
    cms.PSet(detSelection = cms.uint32(221),detLabel = cms.string("FPIXpD1"), binMax = cms.int32(100000)),
    cms.PSet(detSelection = cms.uint32(222),detLabel = cms.string("FPIXpD2"), binMax = cms.int32(100000)),
    cms.PSet(detSelection = cms.uint32(223),detLabel = cms.string("FPIXpD3"), binMax = cms.int32(100000)),
    cms.PSet(detSelection = cms.uint32(2111),detLabel = cms.string("FPIXmD1pan1"), binMax = cms.int32(100000)),
    cms.PSet(detSelection = cms.uint32(2121),detLabel = cms.string("FPIXmD2pan1"), binMax = cms.int32(100000)),
    cms.PSet(detSelection = cms.uint32(2131),detLabel = cms.string("FPIXmD3pan1"), binMax = cms.int32(100000)),
    cms.PSet(detSelection = cms.uint32(2211),detLabel = cms.string("FPIXpD1pan1"), binMax = cms.int32(100000)),
    cms.PSet(detSelection = cms.uint32(2221),detLabel = cms.string("FPIXpD2pan1"), binMax = cms.int32(100000)),
    cms.PSet(detSelection = cms.uint32(2231),detLabel = cms.string("FPIXpD3pan1"), binMax = cms.int32(100000)),
    cms.PSet(detSelection = cms.uint32(2112),detLabel = cms.string("FPIXmD1pan2"), binMax = cms.int32(100000)),
    cms.PSet(detSelection = cms.uint32(2122),detLabel = cms.string("FPIXmD2pan2"), binMax = cms.int32(100000)),
    cms.PSet(detSelection = cms.uint32(2132),detLabel = cms.string("FPIXmD3pan2"), binMax = cms.int32(100000)),
    cms.PSet(detSelection = cms.uint32(2212),detLabel = cms.string("FPIXpD1pan2"), binMax = cms.int32(100000)),
    cms.PSet(detSelection = cms.uint32(2222),detLabel = cms.string("FPIXpD2pan2"), binMax = cms.int32(100000)),
    cms.PSet(detSelection = cms.uint32(2232),detLabel = cms.string("FPIXpD3pan2"), binMax = cms.int32(100000))
    )
process.spclusmultinvestigatornew.digiLumiCorrConfig.wantedSubDets = cms.untracked.VPSet(    
    cms.PSet(detSelection = cms.uint32(101),detLabel = cms.string("BPIX"), binMax = cms.int32(100000)),
    cms.PSet(detSelection = cms.uint32(11),detLabel = cms.string("BPIXL1"), binMax = cms.int32(100000)),
    cms.PSet(detSelection = cms.uint32(12),detLabel = cms.string("BPIXL2"), binMax = cms.int32(100000)),
    cms.PSet(detSelection = cms.uint32(13),detLabel = cms.string("BPIXL3"), binMax = cms.int32(100000)),
    cms.PSet(detSelection = cms.uint32(102),detLabel = cms.string("FPIX"), binMax = cms.int32(100000)),
    cms.PSet(detSelection = cms.uint32(21),detLabel = cms.string("FPIXm"), binMax = cms.int32(100000)),
    cms.PSet(detSelection = cms.uint32(22),detLabel = cms.string("FPIXp"), binMax = cms.int32(100000)),
    cms.PSet(detSelection = cms.uint32(211),detLabel = cms.string("FPIXmD1"), binMax = cms.int32(100000)),
    cms.PSet(detSelection = cms.uint32(212),detLabel = cms.string("FPIXmD2"), binMax = cms.int32(100000)),
    cms.PSet(detSelection = cms.uint32(213),detLabel = cms.string("FPIXmD3"), binMax = cms.int32(100000)),
    cms.PSet(detSelection = cms.uint32(221),detLabel = cms.string("FPIXpD1"), binMax = cms.int32(100000)),
    cms.PSet(detSelection = cms.uint32(222),detLabel = cms.string("FPIXpD2"), binMax = cms.int32(100000)),
    cms.PSet(detSelection = cms.uint32(223),detLabel = cms.string("FPIXpD3"), binMax = cms.int32(100000)),
    cms.PSet(detSelection = cms.uint32(2111),detLabel = cms.string("FPIXmD1pan1"), binMax = cms.int32(100000)),
    cms.PSet(detSelection = cms.uint32(2121),detLabel = cms.string("FPIXmD2pan1"), binMax = cms.int32(100000)),
    cms.PSet(detSelection = cms.uint32(2131),detLabel = cms.string("FPIXmD3pan1"), binMax = cms.int32(100000)),
    cms.PSet(detSelection = cms.uint32(2211),detLabel = cms.string("FPIXpD1pan1"), binMax = cms.int32(100000)),
    cms.PSet(detSelection = cms.uint32(2221),detLabel = cms.string("FPIXpD2pan1"), binMax = cms.int32(100000)),
    cms.PSet(detSelection = cms.uint32(2231),detLabel = cms.string("FPIXpD3pan1"), binMax = cms.int32(100000)),
    cms.PSet(detSelection = cms.uint32(2112),detLabel = cms.string("FPIXmD1pan2"), binMax = cms.int32(100000)),
    cms.PSet(detSelection = cms.uint32(2122),detLabel = cms.string("FPIXmD2pan2"), binMax = cms.int32(100000)),
    cms.PSet(detSelection = cms.uint32(2132),detLabel = cms.string("FPIXmD3pan2"), binMax = cms.int32(100000)),
    cms.PSet(detSelection = cms.uint32(2212),detLabel = cms.string("FPIXpD1pan2"), binMax = cms.int32(100000)),
    cms.PSet(detSelection = cms.uint32(2222),detLabel = cms.string("FPIXpD2pan2"), binMax = cms.int32(100000)),
    cms.PSet(detSelection = cms.uint32(2232),detLabel = cms.string("FPIXpD3pan2"), binMax = cms.int32(100000))
    )
process.spclusmultinvestigatornew.multiplicityMap = cms.InputTag("spclustermultprodnew")

process.load("DPGAnalysis.SiStripTools.multiplicitycorr_cfi")
process.multiplicitycorr.correlationConfigurations = cms.VPSet(
   cms.PSet(xMultiplicityMap = cms.InputTag("spclustermultprodnew"),
            xDetSelection = cms.uint32(12), xDetLabel = cms.string("BPixL2"), xBins = cms.uint32(1000), xMax=cms.double(10000), 
            yMultiplicityMap = cms.InputTag("ssclustermultprodnew"),
            yDetSelection = cms.uint32(103), yDetLabel = cms.string("TIB"), yBins = cms.uint32(1000), yMax=cms.double(50000),
            rBins = cms.uint32(200), scaleFactor = cms.untracked.double(.25),
            runHisto=cms.bool(False),runHistoBXProfile=cms.bool(False),runHistoBX=cms.bool(False),runHisto2D=cms.bool(False)),
   cms.PSet(xMultiplicityMap = cms.InputTag("spclustermultprodnew"),
            xDetSelection = cms.uint32(12), xDetLabel = cms.string("BPixL2"), xBins = cms.uint32(1000), xMax=cms.double(10000), 
            yMultiplicityMap = cms.InputTag("ssclustermultprodnew"),
            yDetSelection = cms.uint32(114), yDetLabel = cms.string("TIDm"), yBins = cms.uint32(1000), yMax=cms.double(20000),
            rBins = cms.uint32(200), scaleFactor = cms.untracked.double(.5),
            runHisto=cms.bool(False),runHistoBXProfile=cms.bool(False),runHistoBX=cms.bool(False),runHisto2D=cms.bool(False)),
   cms.PSet(xMultiplicityMap = cms.InputTag("spclustermultprodnew"),
            xDetSelection = cms.uint32(12), xDetLabel = cms.string("BPixL2"), xBins = cms.uint32(1000), xMax=cms.double(10000), 
            yMultiplicityMap = cms.InputTag("ssclustermultprodnew"),
            yDetSelection = cms.uint32(124), yDetLabel = cms.string("TIDp"), yBins = cms.uint32(1000), yMax=cms.double(20000),
            rBins = cms.uint32(200), scaleFactor = cms.untracked.double(.5),
            runHisto=cms.bool(False),runHistoBXProfile=cms.bool(False),runHistoBX=cms.bool(False),runHisto2D=cms.bool(False)),
   cms.PSet(xMultiplicityMap = cms.InputTag("spclustermultprodnew"),
            xDetSelection = cms.uint32(12), xDetLabel = cms.string("BPixL2"), xBins = cms.uint32(1000), xMax=cms.double(10000), 
            yMultiplicityMap = cms.InputTag("spclustermultprodnew"),
            yDetSelection = cms.uint32(11), yDetLabel = cms.string("BPixL1"), yBins = cms.uint32(1000), yMax=cms.double(10000),
            rBins = cms.uint32(200), scaleFactor = cms.untracked.double(1.),
            runHisto=cms.bool(False),runHistoBXProfile=cms.bool(False),runHistoBX=cms.bool(False),runHisto2D=cms.bool(False))
   )

process.multcorrclustsumm = process.multiplicitycorr.clone()
process.multcorrclustsumm.correlationConfigurations = cms.VPSet(
   cms.PSet(yMultiplicityMap = cms.InputTag("spclustermultprod"),
            yDetSelection = cms.uint32(0), yDetLabel = cms.string("Pixel"), yBins = cms.uint32(1000), yMax=cms.double(1000), 
            xMultiplicityMap = cms.InputTag("clustsummmultprod"),
            xDetSelection = cms.uint32(1005), xDetLabel = cms.string("Pixelcs"), xBins = cms.uint32(1000), xMax=cms.double(1000),
            rBins = cms.uint32(200), scaleFactor = cms.untracked.double(1.),
            runHisto=cms.bool(False),runHistoBXProfile=cms.bool(False),runHistoBX=cms.bool(False),runHisto2D=cms.bool(False)),
   cms.PSet(xMultiplicityMap = cms.InputTag("spclustermultprodnew"),
            xDetSelection = cms.uint32(101), xDetLabel = cms.string("BPIX"), xBins = cms.uint32(1000), xMax=cms.double(1000), 
            yMultiplicityMap = cms.InputTag("clustsummmultprod"),
            yDetSelection = cms.uint32(1007), yDetLabel = cms.string("BPIXcs"), yBins = cms.uint32(1000), yMax=cms.double(1000),
            rBins = cms.uint32(200), scaleFactor = cms.untracked.double(1.),
            runHisto=cms.bool(False),runHistoBXProfile=cms.bool(False),runHistoBX=cms.bool(False),runHisto2D=cms.bool(False)),
   cms.PSet(xMultiplicityMap = cms.InputTag("spclustermultprodnew"),
            xDetSelection = cms.uint32(102), xDetLabel = cms.string("FPIX"), xBins = cms.uint32(1000), xMax=cms.double(1000), 
            yMultiplicityMap = cms.InputTag("clustsummmultprod"),
            yDetSelection = cms.uint32(1006), yDetLabel = cms.string("FPIXcs"), yBins = cms.uint32(1000), yMax=cms.double(1000),
            rBins = cms.uint32(200), scaleFactor = cms.untracked.double(1.),
            runHisto=cms.bool(False),runHistoBXProfile=cms.bool(False),runHistoBX=cms.bool(False),runHisto2D=cms.bool(False)),
   cms.PSet(xMultiplicityMap = cms.InputTag("ssclustermultprod"),
            xDetSelection = cms.uint32(3), xDetLabel = cms.string("TIB"), xBins = cms.uint32(1000), xMax=cms.double(50000), 
            yMultiplicityMap = cms.InputTag("clustsummmultprod"),
            yDetSelection = cms.uint32(1), yDetLabel = cms.string("TIBcs"), yBins = cms.uint32(1000), yMax=cms.double(50000),
            rBins = cms.uint32(200), scaleFactor = cms.untracked.double(1.),
            runHisto=cms.bool(False),runHistoBXProfile=cms.bool(False),runHistoBX=cms.bool(False),runHisto2D=cms.bool(False)),
   cms.PSet(xMultiplicityMap = cms.InputTag("ssclustermultprod"),
            xDetSelection = cms.uint32(4), xDetLabel = cms.string("TID"), xBins = cms.uint32(1000), xMax=cms.double(50000), 
            yMultiplicityMap = cms.InputTag("clustsummmultprod"),
            yDetSelection = cms.uint32(3), yDetLabel = cms.string("TIDcs"), yBins = cms.uint32(1000), yMax=cms.double(50000),
            rBins = cms.uint32(200), scaleFactor = cms.untracked.double(1.),
            runHisto=cms.bool(False),runHistoBXProfile=cms.bool(False),runHistoBX=cms.bool(False),runHisto2D=cms.bool(False)),
   cms.PSet(xMultiplicityMap = cms.InputTag("ssclustermultprod"),
            xDetSelection = cms.uint32(5), xDetLabel = cms.string("TOB"), xBins = cms.uint32(1000), xMax=cms.double(50000), 
            yMultiplicityMap = cms.InputTag("clustsummmultprod"),
            yDetSelection = cms.uint32(2), yDetLabel = cms.string("TOBcs"), yBins = cms.uint32(1000), yMax=cms.double(50000),
            rBins = cms.uint32(200), scaleFactor = cms.untracked.double(1.),
            runHisto=cms.bool(False),runHistoBXProfile=cms.bool(False),runHistoBX=cms.bool(False),runHisto2D=cms.bool(False)),
   cms.PSet(xMultiplicityMap = cms.InputTag("ssclustermultprod"),
            xDetSelection = cms.uint32(6), xDetLabel = cms.string("TEC"), xBins = cms.uint32(1000), xMax=cms.double(50000), 
            yMultiplicityMap = cms.InputTag("clustsummmultprod"),
            yDetSelection = cms.uint32(4), yDetLabel = cms.string("TECcs"), yBins = cms.uint32(1000), yMax=cms.double(50000),
            rBins = cms.uint32(200), scaleFactor = cms.untracked.double(1.),
            runHisto=cms.bool(False),runHistoBXProfile=cms.bool(False),runHistoBX=cms.bool(False),runHisto2D=cms.bool(False))
   )


process.clustsummmultcorr =  cms.EDAnalyzer('MultiplicityCorrelator',
                                            correlationConfigurations = cms.VPSet(
    cms.PSet(xMultiplicityMap = cms.InputTag("clustsummmultprod"), xDetSelection = cms.uint32(0), xDetLabel = cms.string("TK"), xBins = cms.uint32(1000), xMax=cms.double(150000),
             yMultiplicityMap = cms.InputTag("clustsummmultprod"), yDetSelection = cms.uint32(1005), yDetLabel = cms.string("Pixel"), yBins = cms.uint32(1000), yMax=cms.double(20000),
             rBins = cms.uint32(200), scaleFactor = cms.untracked.double(5.),
             runHisto=cms.bool(False),runHistoBXProfile=cms.bool(False),runHistoBX=cms.bool(False),runHisto2D=cms.bool(False))
    )
                                            )
#process.clustsummmultcorr =  cms.EDAnalyzer('MultiplicityCorrelator',
#                                            correlationConfigurations = cms.VPSet(
#    cms.PSet(xMultiplicityMap = cms.InputTag("ssclustermultprod"), xDetSelection = cms.uint32(0), xDetLabel = cms.string("TK"), xBins = cms.uint32(1000), xMax=cms.double(150000),
#             yMultiplicityMap = cms.InputTag("spclustermultprod"), yDetSelection = cms.uint32(0), yDetLabel = cms.string("Pixel"), yBins = cms.uint32(1000), yMax=cms.double(20000),
#             rBins = cms.uint32(200), scaleFactor = cms.untracked.double(5.),
#             runHisto=cms.bool(False),runHistoBXProfile=cms.bool(False),runHistoBX=cms.bool(False),runHisto2D=cms.bool(False))
#    )
#                                            )

process.clustsummmultcorrmanystripclus = process.clustsummmultcorr.clone()

#process.load("TrackingPFG.Utilities.logerroranalyzer_cfi")

process.load("DPGAnalysis.SiStripTools.clusterbigeventsdebugger_cfi")
process.clusterbigeventsdebugger.selections.append(cms.PSet(label=cms.string("Module"),selection=cms.untracked.vstring("0x1fffffff-0x1600546d")))
#process.clusterbigeventsdebugger.collection = cms.InputTag("calZeroBiasClusters")
#process.clusterbigeventsdebugger.foldedStrips = cms.untracked.bool(True)
#process.clusterbigeventsdebugger.singleEvents = cms.bool(True)

process.load("DPGAnalysis.SiStripTools.digibigeventsdebugger_cfi")
process.digibigeventsdebugger.selections = cms.VPSet(
    cms.PSet(label=cms.string("TIB"),selection=cms.untracked.vstring("0x1e000000-0x16000000")),
    cms.PSet(label=cms.string("TIBD"),selection=cms.untracked.vstring("0x1e000000-0x16000000","0x1e000000-0x18000000")),
    cms.PSet(label=cms.string("TECp"),selection=cms.untracked.vstring("0x1e0c0000-0x1c080000")),
    cms.PSet(label=cms.string("TECm"),selection=cms.untracked.vstring("0x1e0c0000-0x1c040000")),
    cms.PSet(label=cms.string("TOB"),selection=cms.untracked.vstring("0x1e000000-0x1a000000")),
    cms.PSet(label=cms.string("TID"),selection=cms.untracked.vstring("0x1e000000-0x18000000"))
    )

process.digibigeventsdebugger.selections.append(cms.PSet(label=cms.string("Module"),selection=cms.untracked.vstring("0x1fffffff-0x1600546d")))
process.digibigeventsdebugger.collection = cms.InputTag("siStripClustersToDigis","ZeroSuppressed")
process.digibigeventsdebugger.foldedStrips = cms.untracked.bool(True)
#process.digibigeventsdebugger.singleEvents = cms.bool(True)

process.load("DPGAnalysis.SiStripTools.fedbadmodulefilter_cfi")
process.fedbadmodulefilter.badModThr = 0
process.fedbadmodulefilter.wantedHisto = cms.untracked.bool(True)
process.fedbadmodulefilter.moduleSelection = cms.untracked.PSet(selection=cms.untracked.vstring("0x1fffffff-0x1600546d"))

process.fedbadmoduleTECmD8 = process.fedbadmodulefilter.clone(moduleSelection = cms.untracked.PSet(selection=cms.untracked.vstring("0x1e0fc000-0x1c060000")))
process.fedbadmoduleTECmD4 = process.fedbadmodulefilter.clone(moduleSelection = cms.untracked.PSet(selection=cms.untracked.vstring("0x1e0fc000-0x1c050000")))
process.fedbadmoduleTECmD7 = process.fedbadmodulefilter.clone(moduleSelection = cms.untracked.PSet(selection=cms.untracked.vstring("0x1e0fc000-0x1c05c000")))

process.spclusmultinvestigatorontrack = process.spclusmultinvestigator.clone(multiplicityMap = cms.InputTag("spclustermultprodontrack"))
process.ssclusmultinvestigatorontrack = process.ssclusmultinvestigator.clone(multiplicityMap = cms.InputTag("ssclustermultprodontrack"))
process.spclusmultinvestigatornewontrack = process.spclusmultinvestigatornew.clone(multiplicityMap = cms.InputTag("spclustermultprodnewontrack"))
process.ssclusmultinvestigatornewontrack = process.ssclusmultinvestigatornew.clone(multiplicityMap = cms.InputTag("ssclustermultprodnewontrack"))

process.seqClusMultInvest = cms.Sequence(
    process.spclusmultinvestigator + process.ssclusmultinvestigator +
    process.spclusmultinvestigatorontrack + process.ssclusmultinvestigatorontrack +
    process.spclusmultinvestigatornew + process.ssclusmultinvestigatornew +
    process.spclusmultinvestigatornewontrack + process.ssclusmultinvestigatornewontrack
                                         + process.multiplicitycorr + process.multcorrclustsumm) 
#process.seqClusMultInvest = cms.Sequence(process.clustsummmultcorr) 
#process.seqClusMultInvest = cms.Sequence(process.ssclusmultinvestigatornew +
#                                         process.clusterbigeventsdebugger + process.digibigeventsdebugger +
#                                         process.fedbadmodulefilter + process.fedbadmoduleTECmD8 + process.fedbadmoduleTECmD4 + process.fedbadmoduleTECmD7 
#                                         ) 



process.p0 = cms.Path(
    #    process.hltSelection +
    process.seqProducers +
    process.seqClusMultInvest)

#process.pmanystripclus = cms.Path(process.seqProducers +
#                                  process.manystripclus53X +
##                                  process.manystripclus +
#                                  process.clustsummmultcorrmanystripclus +
#                                  process.logerroranalyzer
#                                  )

#process.outoddevent = cms.OutputModule("PoolOutputModule",
#                                       fileName = cms.untracked.string("manystripclusevents.root"),
#                                       outputCommands = cms.untracked.vstring("keep *"),
#                                       SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring( "pmanystripclus"))
#                                       )                                       

#process.e = cms.EndPath(process.outoddevent)

#----GlobalTag ------------------------

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = options.globalTag

if options.testTag==1:
    process.stripConditions = cms.ESSource(
        "PoolDBESSource",
        process.CondDBSetup,
        connect = cms.string('frontier://FrontierProd/CMS_COND_31X_STRIP'),
        toGet = cms.VPSet(
        cms.PSet(
        record = cms.string('SiStripNoisesRcd'), tag = cms.string('SiStripNoise_GR10_v2_offline')
        ),
        ),
        )

    process.es_prefer_strips = cms.ESPrefer("PoolDBESSource","stripConditions")

process.TFileService = cms.Service('TFileService',
                                   fileName = cms.string('MultiplicityProducerTest.root')
                                   )

#print process.dumpPython()
