import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing

process = cms.Process("OccupancyPlotsTest")

#prepare options

options = VarParsing.VarParsing("analysis")

options.register ('globalTag',
                  "DONOTEXIST",
                  VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                  VarParsing.VarParsing.varType.string,          # string, int, or float
                  "GlobalTag")
options.register ('HLTprocess',
                  "HLT",
                  VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                  VarParsing.VarParsing.varType.string,          # string, int, or float
                  "HLTProcess")
options.register ('triggerPath',
                  "HLT_*",
                  VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                  VarParsing.VarParsing.varType.string,          # string, int, or float
                  "list of HLT paths")

options.parseArguments()

#

process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True),
    fileMode = cms.untracked.string("FULLMERGE")
    )

process.load("FWCore.MessageService.MessageLogger_cfi")

process.MessageLogger.cout.enable = cms.untracked.bool(True)
process.MessageLogger.cout.threshold = cms.untracked.string("WARNING")
process.MessageLogger.cout.default = cms.untracked.PSet(
    limit = cms.untracked.int32(10000000)
    )
process.MessageLogger.cout.FwkReport = cms.untracked.PSet(
    reportEvery = cms.untracked.int32(10000)
    )

process.MessageLogger.cerr.enable = cms.untracked.bool(True)
process.MessageLogger.cerr.threshold = cms.untracked.string("WARNING")
process.MessageLogger.cerr.default = cms.untracked.PSet(
    limit = cms.untracked.int32(10000000)
    )
process.MessageLogger.cerr.FwkReport = cms.untracked.PSet(
    reportEvery = cms.untracked.int32(100000)
    )

#------------------------------------------------------------------

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.source = cms.Source("PoolSource",
                    fileNames = cms.untracked.vstring(options.inputFiles),
#                    skipBadFiles = cms.untracked.bool(True),
                    inputCommands = cms.untracked.vstring("keep *", "drop *_MEtoEDMConverter_*_*")
                    )

# HLT Selection ------------------------------------------------------------
process.load("HLTrigger.HLTfilters.triggerResultsFilter_cfi")
process.triggerResultsFilter.triggerConditions = cms.vstring(options.triggerPath)
process.triggerResultsFilter.hltResults = cms.InputTag( "TriggerResults", "", options.HLTprocess )
process.triggerResultsFilter.l1tResults = cms.InputTag( "" )
process.triggerResultsFilter.throw = cms.bool(False)

process.seqHLTSelection = cms.Sequence(process.triggerResultsFilter)


#--------------------------------------
#from DPGAnalysis.SiStripTools.occupancyplotsselections_cff import *
from DPGAnalysis.SiStripTools.occupancyplotsselections_simplified_cff import *

process.ssclusmultprod = cms.EDProducer("SiStripClusterMultiplicityProducer",
                                        clusterdigiCollection = cms.InputTag("siStripClusters"),
                                        wantedSubDets = cms.VPSet()
                                        )
process.ssclusmultprod.wantedSubDets.extend(OccupancyPlotsStripWantedSubDets)

process.ssclusoccuprod = cms.EDProducer("SiStripClusterMultiplicityProducer",
                                        clusterdigiCollection = cms.InputTag("siStripClusters"),
                                        withClusterSize = cms.untracked.bool(True),
                                        wantedSubDets = cms.VPSet()
                                        )
process.ssclusoccuprod.wantedSubDets.extend(OccupancyPlotsStripWantedSubDets)

process.spclusmultprod = cms.EDProducer("SiPixelClusterMultiplicityProducer",
                                        clusterdigiCollection = cms.InputTag("siPixelClusters"),
                                        wantedSubDets = cms.VPSet()
                                        )
process.spclusmultprod.wantedSubDets.extend(OccupancyPlotsPixelWantedSubDets)
process.spclusmultprod.wantedSubDets.extend(cms.VPSet(
    cms.PSet(detSelection = cms.uint32(0),detLabel = cms.string("Pixel")),
    cms.PSet(detSelection = cms.uint32(1),detLabel = cms.string("BPIX")),
    cms.PSet(detSelection = cms.uint32(2),detLabel = cms.string("FPIX")),
    cms.PSet(detSelection=cms.uint32(11),detLabel=cms.string("BPIX_L1"),selection=cms.untracked.vstring("0x1e0f0000-0x12010000")),
    cms.PSet(detSelection=cms.uint32(12),detLabel=cms.string("BPIX_L2"),selection=cms.untracked.vstring("0x1e0f0000-0x12020000")),
    cms.PSet(detSelection=cms.uint32(13),detLabel=cms.string("BPIX_L3"),selection=cms.untracked.vstring("0x1e0f0000-0x12030000")),
    cms.PSet(detSelection=cms.uint32(21),detLabel=cms.string("FPIX_m"),selection=cms.untracked.vstring("0x1f800000-0x14800000")),
    cms.PSet(detSelection=cms.uint32(22),detLabel=cms.string("FPIX_p"),selection=cms.untracked.vstring("0x1f800000-0x15000000")),
    cms.PSet(detSelection=cms.uint32(99),detLabel=cms.string("Lumi"),selection=cms.untracked.vstring("0x1e0f0000-0x12020000",
                                                                                                     "0x1e0f0000-0x12030000",
                                                                                                     "0x1f800000-0x14800000",
                                                                                                     "0x1f800000-0x15000000"))
))

process.spclusoccuprod = cms.EDProducer("SiPixelClusterMultiplicityProducer",
                                        clusterdigiCollection = cms.InputTag("siPixelClusters"),
                                        withClusterSize = cms.untracked.bool(True),
                                        wantedSubDets = cms.VPSet()
                                        )
process.spclusoccuprod.wantedSubDets.extend(OccupancyPlotsPixelWantedSubDets)

process.seqMultProd = cms.Sequence(process.ssclusmultprod + process.ssclusoccuprod +
                                   process.spclusmultprod + process.spclusoccuprod)

process.load("DPGAnalysis.SiStripTools.occupancyplots_cfi")
process.occupancyplots.wantedSubDets = OccupancyPlotsStripWantedSubDets

process.pixeloccupancyplots = process.occupancyplots.clone()
process.pixeloccupancyplots.wantedSubDets = cms.VPSet()
process.pixeloccupancyplots.wantedSubDets.extend(OccupancyPlotsPixelWantedSubDets)
process.pixeloccupancyplots.multiplicityMaps = cms.VInputTag(cms.InputTag("spclusmultprod"))
process.pixeloccupancyplots.occupancyMaps = cms.VInputTag(cms.InputTag("spclusoccuprod"))

process.alloccupancyplots = process.occupancyplots.clone()
process.alloccupancyplots.wantedSubDets = cms.VPSet()
process.alloccupancyplots.wantedSubDets.extend(OccupancyPlotsPixelWantedSubDets)
process.alloccupancyplots.wantedSubDets.extend(OccupancyPlotsStripWantedSubDets)
process.alloccupancyplots.multiplicityMaps = cms.VInputTag(cms.InputTag("spclusmultprod"),cms.InputTag("ssclusmultprod"))
process.alloccupancyplots.occupancyMaps = cms.VInputTag(cms.InputTag("spclusoccuprod"),cms.InputTag("ssclusoccuprod"))

process.load("DPGAnalysis.SiStripTools.spclusmultvtxposcorr_cfi")
process.spclusmultvtxposcorr.multiplicityMap = cms.InputTag("spclusmultprod")
process.spclusmultvtxposcorr.digiVtxPosCorrConfig.wantedSubDets = cms.untracked.VPSet(    
    cms.PSet(detSelection = cms.uint32(0),detLabel = cms.string("Pixel"), binMax = cms.int32(200000)),
    cms.PSet(detSelection = cms.uint32(1),detLabel = cms.string("BPIX"), binMax = cms.int32(200000)),
    cms.PSet(detSelection = cms.uint32(11),detLabel = cms.string("BPIX_L1"), binMax = cms.int32(200000)),
    cms.PSet(detSelection = cms.uint32(12),detLabel = cms.string("BPIX_L2"), binMax = cms.int32(200000)),
    cms.PSet(detSelection = cms.uint32(13),detLabel = cms.string("BPIX_L3"), binMax = cms.int32(200000)),
    cms.PSet(detSelection = cms.uint32(2),detLabel = cms.string("FPIX"), binMax = cms.int32(200000)),
    cms.PSet(detSelection = cms.uint32(21),detLabel = cms.string("FPIX_m"), binMax = cms.int32(200000)),
    cms.PSet(detSelection = cms.uint32(22),detLabel = cms.string("FPIX_p"), binMax = cms.int32(200000)),
    cms.PSet(detSelection = cms.uint32(99),detLabel = cms.string("Lumi"), binMax = cms.int32(200000)),
    cms.PSet(detSelection = cms.uint32(111),detLabel = cms.string("BPIX_L1_mod_1"), binMax = cms.int32(200000)),
    cms.PSet(detSelection = cms.uint32(112),detLabel = cms.string("BPIX_L1_mod_2"), binMax = cms.int32(200000)),
    cms.PSet(detSelection = cms.uint32(113),detLabel = cms.string("BPIX_L1_mod_3"), binMax = cms.int32(200000)),
    cms.PSet(detSelection = cms.uint32(114),detLabel = cms.string("BPIX_L1_mod_4"), binMax = cms.int32(200000)),
    cms.PSet(detSelection = cms.uint32(115),detLabel = cms.string("BPIX_L1_mod_5"), binMax = cms.int32(200000)),
    cms.PSet(detSelection = cms.uint32(116),detLabel = cms.string("BPIX_L1_mod_6"), binMax = cms.int32(200000)),
    cms.PSet(detSelection = cms.uint32(117),detLabel = cms.string("BPIX_L1_mod_7"), binMax = cms.int32(200000)),
    cms.PSet(detSelection = cms.uint32(118),detLabel = cms.string("BPIX_L1_mod_8"), binMax = cms.int32(200000))
    )


process.load("TrackingPFG.Utilities.bxlumianalyzer_cfi")
process.load("Validation.RecoVertex.mcverticesanalyzer_cfi")

process.goodVertices = cms.EDFilter("VertexSelector",
   src = cms.InputTag("offlinePrimaryVertices"),
   cut = cms.string("!isFake && ndof > 4 && abs(z) <= 24 && position.Rho <= 2"),  
   filter = cms.bool(False),   # otherwise it won't filter the events, just produce an empty vertex collection.
)

process.load("Validation.RecoVertex.anotherprimaryvertexanalyzer_cfi")
process.primaryvertexanalyzer.pvCollection=cms.InputTag("goodVertices")
process.primaryvertexanalyzer.vHistogramMakerPSet.runHisto=cms.untracked.bool(False)
process.primaryvertexanalyzer.vHistogramMakerPSet.runHistoProfile=cms.untracked.bool(False)
process.primaryvertexanalyzer.vHistogramMakerPSet.runHistoBXProfile=cms.untracked.bool(False)

process.seqAnalyzers = cms.Sequence(process.bxlumianalyzer + process.goodVertices + process.primaryvertexanalyzer +
                                    process.occupancyplots + process.pixeloccupancyplots + process.alloccupancyplots +
                                    process.spclusmultvtxposcorr + process.mcverticesanalyzer ) 

#-------------------------------------------------------------------------------------------

process.seqProducers = cms.Sequence(process.seqMultProd)

process.p0 = cms.Path(
    process.seqHLTSelection +
    process.seqProducers +
    process.seqAnalyzers
    )

#----GlobalTag ------------------------

process.load("Configuration.StandardSequences.GeometryDB_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, options.globalTag, '')



process.siStripQualityESProducer.ListOfRecordToMerge=cms.VPSet(
#    cms.PSet( record = cms.string("SiStripDetVOffRcd"),    tag    = cms.string("") ),
    cms.PSet( record = cms.string("SiStripDetCablingRcd"), tag    = cms.string("") ),
    cms.PSet( record = cms.string("RunInfoRcd"),           tag    = cms.string("") ),
    cms.PSet( record = cms.string("SiStripBadChannelRcd"), tag    = cms.string("") ),
    cms.PSet( record = cms.string("SiStripBadFiberRcd"),   tag    = cms.string("") ),
    cms.PSet( record = cms.string("SiStripBadModuleRcd"),  tag    = cms.string("") )
)

process.TFileService = cms.Service('TFileService',
#                                   fileName = cms.string('OccupancyPlotsTest_newschema.root')
                                   fileName = cms.string('OccupancyPlotsTest_vtxpos.root')
                                   )


#print process.dumpPython()
