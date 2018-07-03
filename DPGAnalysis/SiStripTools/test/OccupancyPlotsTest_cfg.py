import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing

from PhysicsTools.PatAlgos.tools.helpers import cloneProcessingSnippet

process = cms.Process("OccupancyPlotsTest")

#prepare options

options = VarParsing.VarParsing("analysis")

options.register ('globalTag',
                  "DONOTEXIST",
                  VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                  VarParsing.VarParsing.varType.string,          # string, int, or float
                  "GlobalTag")
options.register ('fromRAW',
                  "0",
                  VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                  VarParsing.VarParsing.varType.int,          # string, int, or float
                  "=1 if from RAW")
options.register ('withTracks',
                  "0",
                  VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                  VarParsing.VarParsing.varType.int,          # string, int, or float
                  "=1 if analysis of on-track clusters has to be done")
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
options.register ('trackCollection',
                  "generalTracks",
                  VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                  VarParsing.VarParsing.varType.string,          # string, int, or float
                  "Track collection to use")


options.parseArguments()

#

process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True),
    fileMode = cms.untracked.string("FULLMERGE")
    )

process.load("FWCore.MessageService.MessageLogger_cfi")

process.MessageLogger.destinations.extend(cms.vstring("detids"))
process.MessageLogger.categories.extend(cms.vstring("GeometricDetBuilding","DuplicateHitFinder","BuildingTrackerDetId",
                                                    "SubDetectorGeometricDetType","BuildingGeomDetUnits","LookingForFirstStrip",
                                                    "BuildingSubDetTypeMap","SubDetTypeMapContent","NumberOfLayers","IsThereTest"))
process.MessageLogger.cout.placeholder = cms.untracked.bool(False)
process.MessageLogger.cout.threshold = cms.untracked.string("INFO")
#process.MessageLogger.cout.threshold = cms.untracked.string("WARNING")
#process.MessageLogger.debugModules = cms.untracked.vstring("*")
process.MessageLogger.cout.default = cms.untracked.PSet(
    limit = cms.untracked.int32(0)
    )
process.MessageLogger.detids = cms.untracked.PSet(
    default = cms.untracked.PSet(
        limit = cms.untracked.int32(0)
        ),
    BuildingTrackerDetId = cms.untracked.PSet(
        limit = cms.untracked.int32(100000000)
        ),
    GeometricDetBuilding = cms.untracked.PSet(
        limit = cms.untracked.int32(100000000)
        ),
    SubDetectorGeometricDetType = cms.untracked.PSet(
        limit = cms.untracked.int32(100000000)
        ),
    BuildingGeomDetUnits = cms.untracked.PSet(
        limit = cms.untracked.int32(100000000)
        ),
    LookingForFirstStrip = cms.untracked.PSet(
        limit = cms.untracked.int32(100000000)
        ),
    BuildingSubDetTypeMap = cms.untracked.PSet(
        limit = cms.untracked.int32(100000000)
        ),
    SubDetTypeMapContent = cms.untracked.PSet(
        limit = cms.untracked.int32(100000000)
        ),
    NumberOfLayers = cms.untracked.PSet(
        limit = cms.untracked.int32(100000000)
        ),
    IsThereTest = cms.untracked.PSet(
        limit = cms.untracked.int32(100000000)
        ),
    threshold = cms.untracked.string("DEBUG")
    )    
process.MessageLogger.cout.DuplicateHitFinder = cms.untracked.PSet(
    limit = cms.untracked.int32(100000000)
    )
process.MessageLogger.cout.FwkSummary = cms.untracked.PSet(
    limit = cms.untracked.int32(100000000)
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

#------------------------------------------------------------------

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(options.maxEvents) )

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
if options.triggerPath=="*":
    process.seqHLTSelection = cms.Sequence()


#--------------------------------------
process.load('Configuration.Geometry.GeometryExtended2015Reco_cff')
process.load('Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff')
process.load("Configuration.StandardSequences.Reconstruction_cff")

process.seqRECO = cms.Sequence()

if options.fromRAW == 1:
    process.load("Configuration.StandardSequences.RawToDigi_Data_cff")
    process.load("Configuration.StandardSequences.L1Reco_cff")
    process.siPixelClusters = process.siPixelClustersPreSplitting.clone()
    process.seqRECO = cms.Sequence(process.scalersRawToDigi +
                                   process.siStripDigis + process.siStripZeroSuppression + process.siStripClusters
                                   + process.siPixelDigis + process.siPixelClusters )


#

process.froml1abcHEs = cms.EDProducer("EventWithHistoryProducerFromL1ABC",
                                      l1ABCCollection=cms.InputTag("scalersRawToDigi")
                                      )
process.load("DPGAnalysis.SiStripTools.apvcyclephaseproducerfroml1tsDB_cfi")
process.load("DPGAnalysis.SiStripTools.eventtimedistribution_cfi")
process.eventtimedistribution.historyProduct = cms.InputTag("froml1abcHEs")

process.seqEventHistoryReco = cms.Sequence(process.froml1abcHEs + process.APVPhases)
process.seqEventHistory = cms.Sequence(process.eventtimedistribution)

#from DPGAnalysis.SiStripTools.occupancyplotsselections_cff import *
from DPGAnalysis.SiStripTools.occupancyplotsselections_simplified_cff import *

process.ssclusmultprod = cms.EDProducer("SiStripClusterMultiplicityProducer",
                                        clusterdigiCollection = cms.InputTag("siStripClusters"),
                                        wantedSubDets = cms.VPSet(
        cms.PSet(detSelection = cms.uint32(0),detLabel = cms.string("TK")),
        cms.PSet(detSelection = cms.uint32(3),detLabel = cms.string("TIB")),
        cms.PSet(detSelection = cms.uint32(4),detLabel = cms.string("TID")),
        cms.PSet(detSelection = cms.uint32(5),detLabel = cms.string("TOB")),
        cms.PSet(detSelection = cms.uint32(6),detLabel = cms.string("TEC"))
        )
                                        )
process.ssclusmultprod.wantedSubDets.extend(OccupancyPlotsStripWantedSubDets)
process.ssclusmultprodontrack=process.ssclusmultprod.clone(clusterdigiCollection = cms.InputTag("AlignmentTrackSelector"))

process.ssclusoccuprod = cms.EDProducer("SiStripClusterMultiplicityProducer",
                                        clusterdigiCollection = cms.InputTag("siStripClusters"),
                                        withClusterSize = cms.untracked.bool(True),
                                        wantedSubDets = cms.VPSet()
                                        )
process.ssclusoccuprod.wantedSubDets.extend(OccupancyPlotsStripWantedSubDets)
process.ssclusoccuprodontrack=process.ssclusoccuprod.clone(clusterdigiCollection = cms.InputTag("AlignmentTrackSelector"))

process.spclusmultprod = cms.EDProducer("SiPixelClusterMultiplicityProducer",
                                        clusterdigiCollection = cms.InputTag("siPixelClusters"),
                                        wantedSubDets = cms.VPSet(
        cms.PSet(detSelection = cms.uint32(0),detLabel = cms.string("Pixel")),
        cms.PSet(detSelection = cms.uint32(1),detLabel = cms.string("BPIX")),
        cms.PSet(detSelection = cms.uint32(2),detLabel = cms.string("FPIX"))
        )
                                        )
process.spclusmultprod.wantedSubDets.extend(OccupancyPlotsPixelWantedSubDets)
process.spclusmultprodontrack=process.spclusmultprod.clone(clusterdigiCollection = cms.InputTag("AlignmentTrackSelector"))

process.spclusoccuprod = cms.EDProducer("SiPixelClusterMultiplicityProducer",
                                        clusterdigiCollection = cms.InputTag("siPixelClusters"),
                                        withClusterSize = cms.untracked.bool(True),
                                        wantedSubDets = cms.VPSet()
                                        )
process.spclusoccuprod.wantedSubDets.extend(OccupancyPlotsPixelWantedSubDets)
process.spclusoccuprodontrack=process.spclusoccuprod.clone(clusterdigiCollection = cms.InputTag("AlignmentTrackSelector"))

process.seqMultProd = cms.Sequence(process.ssclusmultprod + process.ssclusoccuprod +
                                   process.spclusmultprod + process.spclusoccuprod)

if options.withTracks == 1:
    process.seqMultProd = cms.Sequence(process.ssclusmultprod + process.ssclusoccuprod +
                                       process.spclusmultprod + process.spclusoccuprod +
                                       process.ssclusmultprodontrack + process.ssclusoccuprodontrack +
                                       process.spclusmultprodontrack + process.spclusoccuprodontrack )


process.load("DPGAnalysis.SiStripTools.ssclusmultinvestigator_cfi")
process.ssclusmultinvestigator.multiplicityMap=cms.InputTag("ssclusmultprod")
process.ssclusmultinvestigator.scaleFactor=cms.untracked.int32(1)
process.load("DPGAnalysis.SiStripTools.spclusmultinvestigator_cfi")
process.spclusmultinvestigator.multiplicityMap=cms.InputTag("spclusmultprod")
process.spclusmultinvestigator.scaleFactor=cms.untracked.int32(10)


process.load("DPGAnalysis.SiStripTools.occupancyplots_cfi")
process.occupancyplots.wantedSubDets = process.ssclusoccuprod.wantedSubDets

process.occupancyplotsontrack = process.occupancyplots.clone()
process.occupancyplotsontrack.wantedSubDets = process.ssclusoccuprodontrack.wantedSubDets
process.occupancyplotsontrack.multiplicityMaps = cms.VInputTag(cms.InputTag("ssclusmultprodontrack"))
process.occupancyplotsontrack.occupancyMaps = cms.VInputTag(cms.InputTag("ssclusoccuprodontrack"))

process.pixeloccupancyplots = process.occupancyplots.clone()
process.pixeloccupancyplots.wantedSubDets = process.spclusoccuprod.wantedSubDets
process.pixeloccupancyplots.multiplicityMaps = cms.VInputTag(cms.InputTag("spclusmultprod"))
process.pixeloccupancyplots.occupancyMaps = cms.VInputTag(cms.InputTag("spclusoccuprod"))

process.pixeloccupancyplotsontrack = process.occupancyplots.clone()
process.pixeloccupancyplotsontrack.wantedSubDets = process.spclusoccuprodontrack.wantedSubDets
process.pixeloccupancyplotsontrack.multiplicityMaps = cms.VInputTag(cms.InputTag("spclusmultprodontrack"))
process.pixeloccupancyplotsontrack.occupancyMaps = cms.VInputTag(cms.InputTag("spclusoccuprodontrack"))

process.alloccupancyplots = process.occupancyplots.clone()
process.alloccupancyplots.wantedSubDets = cms.VPSet()
process.alloccupancyplots.wantedSubDets.extend(OccupancyPlotsPixelWantedSubDets)
process.alloccupancyplots.wantedSubDets.extend(OccupancyPlotsStripWantedSubDets)
process.alloccupancyplots.multiplicityMaps = cms.VInputTag(cms.InputTag("spclusmultprod"),cms.InputTag("ssclusmultprod"))
process.alloccupancyplots.occupancyMaps = cms.VInputTag(cms.InputTag("spclusoccuprod"),cms.InputTag("ssclusoccuprod"))

process.alloccupancyplotsontrack = process.occupancyplots.clone()
process.alloccupancyplotsontrack.wantedSubDets = cms.VPSet()
process.alloccupancyplotsontrack.wantedSubDets.extend(OccupancyPlotsPixelWantedSubDets)
process.alloccupancyplotsontrack.wantedSubDets.extend(OccupancyPlotsStripWantedSubDets)
process.alloccupancyplotsontrack.multiplicityMaps = cms.VInputTag(cms.InputTag("spclusmultprodontrack"),cms.InputTag("ssclusmultprodontrack"))
process.alloccupancyplotsontrack.occupancyMaps = cms.VInputTag(cms.InputTag("spclusoccuprodontrack"),cms.InputTag("ssclusoccuprodontrack"))

#process.layersoccupancyplots = process.occupancyplots.clone()
#process.layersoccupancyplots.wantedSubDets = cms.VPSet()
#process.layersoccupancyplots.wantedSubDets.extend(OccupancyPlotsPixelWantedLayers)
#process.layersoccupancyplots.wantedSubDets.extend(OccupancyPlotsStripWantedLayers)
#process.layersoccupancyplots.multiplicityMaps = cms.VInputTag(cms.InputTag("spclusmultprodontrack"),cms.InputTag("ssclusmultprodontrack"))
#process.layersoccupancyplots.occupancyMaps = cms.VInputTag(cms.InputTag("spclusoccuprodontrack"),cms.InputTag("ssclusoccuprodontrack"))

#process.layersoccupancyplotsontrack = process.occupancyplots.clone()
#process.layersoccupancyplotsontrack.wantedSubDets = cms.VPSet()
#process.layersoccupancyplotsontrack.wantedSubDets.extend(OccupancyPlotsPixelWantedLayers)
#process.layersoccupancyplotsontrack.wantedSubDets.extend(OccupancyPlotsStripWantedLayers)
#process.layersoccupancyplotsontrack.multiplicityMaps = cms.VInputTag(cms.InputTag("spclusmultprod"),cms.InputTag("ssclusmultprod"))
#process.layersoccupancyplotsontrack.occupancyMaps = cms.VInputTag(cms.InputTag("spclusoccuprod"),cms.InputTag("ssclusoccuprod"))


#process.load("TrackingPFG.Utilities.bxlumianalyzer_cfi")

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

process.load("DPGAnalysis.SiStripTools.trackcount_cfi")
process.trackcount.trackCollection = cms.InputTag(options.trackCollection)

process.load("DPGAnalysis.SiStripTools.duplicaterechits_cfi")
process.duplicaterechits.trackCollection = cms.InputTag(options.trackCollection)

process.seqAnalyzers = cms.Sequence(
    process.seqEventHistory +
    process.spclusmultinvestigator + process.ssclusmultinvestigator +
    process.occupancyplots +
    process.pixeloccupancyplots +
    process.alloccupancyplots)

if options.withTracks == 1:
    process.seqAnalyzers = cms.Sequence(
        process.seqEventHistory +
        #process.bxlumianalyzer + 
        process.primaryvertexanalyzer +
        process.spclusmultinvestigator + process.ssclusmultinvestigator +
        process.occupancyplots +     process.occupancyplotsontrack + 
        process.pixeloccupancyplots +     process.pixeloccupancyplotsontrack + 
        process.alloccupancyplots +     process.alloccupancyplotsontrack +
        process.trackcount 
        # + process.duplicaterechits
) 

#-------------------------------------------------------------------------------------------

process.load("Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi")

process.seqProducers = cms.Sequence(process.seqEventHistoryReco + process.seqMultProd)

if options.withTracks == 1:
    process.seqProducers = cms.Sequence(process.seqEventHistoryReco + 
                                        process.AlignmentTrackSelector + 
                                        process.goodVertices +
                                        process.seqMultProd)

#----GlobalTag ------------------------

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

process.SiStripDetInfoFileReader = cms.Service("SiStripDetInfoFileReader")

process.TFileService = cms.Service('TFileService',
                                   fileName = cms.string('OccupancyPlotsTest_'+options.tag+'.root')
                                   )

cloneProcessingSnippet(process,process.seqAnalyzers,"All")

process.p0 = cms.Path(
    process.seqRECO +
    process.seqProducers +
    process.seqAnalyzersAll +
    process.seqHLTSelection +
    process.seqAnalyzers
)


#print process.dumpPython()
