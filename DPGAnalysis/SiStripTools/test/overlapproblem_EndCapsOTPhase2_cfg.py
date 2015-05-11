import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing

process = cms.Process("OverlapProblem")

#prepare options

options = VarParsing.VarParsing("analysis")

options.register('globalTag',
                 "DONOTEXIST",
                 VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                 VarParsing.VarParsing.varType.string,          # string, int, or float
                 "GlobalTag")

options.parseArguments()

#
process.load("DPGAnalysis.SiStripTools.processOptions_cff")
process.load("DPGAnalysis.SiStripTools.MessageLogger_cff")

process.MessageLogger.destinations.extend(cms.vstring("tkdetlayers"))
process.MessageLogger.categories.extend(cms.vstring("NoCluster","ClusterFound","TkDetLayers","DiskNames",
                                                    "BuildingPixelForwardLayer","BuildingPhase2OTECRingedLayer",
                                                    "BuildingPixelBarrel","BuildingPixelBarrelLayer","BuildingPhase2OTBarrelLayer","Phase2OTBarrelRodRadii"))
process.MessageLogger.cout.placeholder = cms.untracked.bool(False)
#process.MessageLogger.cout.threshold = cms.untracked.string("INFO")
process.MessageLogger.cout.threshold = cms.untracked.string("DEBUG")
process.MessageLogger.debugModules = cms.untracked.vstring("*")
process.MessageLogger.tkdetlayers = cms.untracked.PSet (
    threshold = cms.untracked.string("INFO"),
    default = cms.untracked.PSet(limit = cms.untracked.int32(0)),
    TkDetLayers = cms.untracked.PSet(limit = cms.untracked.int32(100000))
    )
process.MessageLogger.cout.default = cms.untracked.PSet(
    limit = cms.untracked.int32(0)
    )
process.MessageLogger.cout.FwkSummary = cms.untracked.PSet(
    limit = cms.untracked.int32(100000000)
    )
process.MessageLogger.cout.FwkReport = cms.untracked.PSet(
    reportEvery = cms.untracked.int32(1000)
    )
process.MessageLogger.cout.NoCluster = cms.untracked.PSet(
    limit = cms.untracked.int32(100000)
    )
process.MessageLogger.cout.DiskNames = cms.untracked.PSet(
    limit = cms.untracked.int32(100000)
    )
process.MessageLogger.cout.ClusterFound = cms.untracked.PSet(
    limit = cms.untracked.int32(100000)
    )
process.MessageLogger.cout.BuildingPixelForwardLayer = cms.untracked.PSet(
    limit = cms.untracked.int32(100000)
    )
process.MessageLogger.cout.BuildingPhase2OTECRingedLayer = cms.untracked.PSet(
    limit = cms.untracked.int32(100000)
    )
process.MessageLogger.cout.BuildingPhase2OTBarrelLayer = cms.untracked.PSet(
    limit = cms.untracked.int32(100000)
    )
process.MessageLogger.cout.BuildingPixelBarrelLayer = cms.untracked.PSet(
    limit = cms.untracked.int32(100000)
    )
process.MessageLogger.cout.BuildingPixelBarrel = cms.untracked.PSet(
    limit = cms.untracked.int32(100000)
    )
process.MessageLogger.cout.Phase2OTBarrelRodRadii = cms.untracked.PSet(
    limit = cms.untracked.int32(100000)
    )

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(options.maxEvents) )

process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(options.inputFiles),
                            #                    skipBadFiles = cms.untracked.bool(True),
                            inputCommands = cms.untracked.vstring("keep *", "drop *_MEtoEDMConverter_*_*")
                            )

#process.load("Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff")
#process.load("Configuration.StandardSequences.GeometryDB_cff")
process.load('Configuration.Geometry.GeometryExtended2023MuonReco_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_PostLS1_cff')
#process.load("Configuration.Geometry.GeometryExtendedPhaseIPixelReco_cff")
#process.load("Configuration.Geometry.GeometryExtendedPhaseIPixel_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")



process.load("SimTracker.TrackAssociatorProducers.trackAssociatorByHits_cfi")


from SLHCUpgradeSimulations.Configuration.phase2TkCustomsBE5DPixel10D import *

process.load("RecoTracker.TrackProducer.TrackRefitters_cff")
process.refittedTracks = process.TrackRefitter.clone(src = cms.InputTag("generalTracks"))
#process.refittedTracks.src = cms.InputTag("AlignmentTrackSelector")
process.refittedTracks.TTRHBuilder = cms.string('WithTrackAngle')

# Need these until pixel templates are used
process.load("SLHCUpgradeSimulations.Geometry.recoFromSimDigis_cff")
# PixelCPEGeneric #
process.PixelCPEGenericESProducer.Upgrade = cms.bool(True)
process.PixelCPEGenericESProducer.UseErrorsFromTemplates = cms.bool(False)
process.PixelCPEGenericESProducer.LoadTemplatesFromDB = cms.bool(False)
process.PixelCPEGenericESProducer.TruncatePixelCharge = cms.bool(False)
process.PixelCPEGenericESProducer.IrradiationBiasCorrection = False
process.PixelCPEGenericESProducer.DoCosmics = False

process.seqTrackRefitting = cms.Sequence(process.refittedTracks)

#process.KFFittingSmootherWithOutliersRejectionAndRK.LogPixelProbabilityCut = cms.double(-16.0)


process.load("DPGAnalysis.SiStripTools.overlapproblemtsosanalyzer_cfi")
process.overlapproblemtsosanalyzer.tsosHMConf.wanted2DHistos = cms.untracked.bool(True)

from DPGAnalysis.SiStripTools.occupancyplotsselections_phase2_cff import *

process.overlapproblemtsosanalyzer.tsosHMConf.wantedSubDets = OccupancyPlotsTECxWantedSubDets
process.overlapproblemtsosanalyzer.tsosHMConf.wantedSubDets.extend(OccupancyPlotsTECxOddEvenWantedSubDets)
process.overlapproblemtsosanalyzer.tsosHMConf.wantedSubDets.extend(OccupancyPlotsFPIXR1WantedSubDets)
process.overlapproblemtsosanalyzer.tsosHMConf.wantedSubDets.extend(OccupancyPlotsFPIXR2WantedSubDets)
process.overlapproblemtsosanalyzer.tsosHMConf.wantedSubDets.extend(OccupancyPlotsFPIXmDetailedWantedSubDets)
process.overlapproblemtsosanalyzer.tsosHMConf.wantedSubDets.extend(OccupancyPlotsFPIXpDetailedWantedSubDets)

process.overlapproblemtsosall = process.overlapproblemtsosanalyzer.clone()
process.overlapproblemtsosall.tsosHMConf.wantedSubDets = OccupancyPlotsPixelWantedSubDets

#process.overlapproblemtsosanalyzer.debugMode = cms.untracked.bool(True)
    
process.spclusmultprod = cms.EDProducer("SiPixelClusterMultiplicityProducer",
                                        clusterdigiCollection = cms.InputTag("siPixelClusters"),
                                        wantedSubDets = cms.VPSet()
                                        )
process.spclusmultprod.wantedSubDets.extend(OccupancyPlotsPixelWantedSubDets)
process.spclusmultprodontrack=process.spclusmultprod.clone(clusterdigiCollection = cms.InputTag("AlignmentTrackSelector"))

process.spclusmultprodxy = process.spclusmultprod.clone()
process.spclusmultprodxy.wantedSubDets = OccupancyPlotsFPIXmDetailedWantedSubDets
process.spclusmultprodxy.wantedSubDets.extend(OccupancyPlotsFPIXpDetailedWantedSubDets)
process.spclusmultprodxyontrack=process.spclusmultprodxy.clone(clusterdigiCollection = cms.InputTag("AlignmentTrackSelector"))

process.spclusoccuprod = cms.EDProducer("SiPixelClusterMultiplicityProducer",
                                        clusterdigiCollection = cms.InputTag("siPixelClusters"),
                                        withClusterSize = cms.untracked.bool(True),
                                        wantedSubDets = cms.VPSet()
                                        )
process.spclusoccuprod.wantedSubDets.extend(OccupancyPlotsPixelWantedSubDets)
process.spclusoccuprodontrack=process.spclusoccuprod.clone(clusterdigiCollection = cms.InputTag("AlignmentTrackSelector"))

process.spclusoccuprodxy = process.spclusoccuprod.clone()
process.spclusoccuprodxy.wantedSubDets = OccupancyPlotsFPIXmDetailedWantedSubDets
process.spclusoccuprodxy.wantedSubDets.extend(OccupancyPlotsFPIXpDetailedWantedSubDets)
process.spclusoccuprodxyontrack=process.spclusoccuprodxy.clone(clusterdigiCollection = cms.InputTag("AlignmentTrackSelector"))

process.seqMultProd = cms.Sequence(
                                   process.spclusmultprod + process.spclusoccuprod +
                                   process.spclusmultprodontrack + process.spclusoccuprodontrack +
                                   process.spclusmultprodxy + process.spclusoccuprodxy +
                                   process.spclusmultprodxyontrack + process.spclusoccuprodxyontrack 
                                   )

process.load("DPGAnalysis.SiStripTools.occupancyplots_cfi")
process.occupancyplots.file = cms.untracked.FileInPath("SLHCUpgradeSimulations/Geometry/data/PhaseII/Pixel10D/PixelSkimmedGeometry.txt")

process.pixeloccupancyplots = process.occupancyplots.clone()
process.pixeloccupancyplots.wantedSubDets = process.spclusmultprod.wantedSubDets
process.pixeloccupancyplots.multiplicityMaps = cms.VInputTag(cms.InputTag("spclusmultprod"))
process.pixeloccupancyplots.occupancyMaps = cms.VInputTag(cms.InputTag("spclusoccuprod"))

process.pixeloccupancyxyplots = process.occupancyplots.clone()
process.pixeloccupancyxyplots.wantedSubDets = process.spclusmultprodxy.wantedSubDets
process.pixeloccupancyxyplots.multiplicityMaps = cms.VInputTag(cms.InputTag("spclusmultprodxy"))
process.pixeloccupancyxyplots.occupancyMaps = cms.VInputTag(cms.InputTag("spclusoccuprodxy"))

process.pixeloccupancyplotsontrack = process.pixeloccupancyplots.clone()
process.pixeloccupancyplotsontrack.wantedSubDets = process.spclusmultprodontrack.wantedSubDets
process.pixeloccupancyplotsontrack.multiplicityMaps = cms.VInputTag(cms.InputTag("spclusmultprodontrack"))
process.pixeloccupancyplotsontrack.occupancyMaps = cms.VInputTag(cms.InputTag("spclusoccuprodontrack"))

process.pixeloccupancyxyplotsontrack = process.pixeloccupancyxyplots.clone()
process.pixeloccupancyxyplotsontrack.wantedSubDets = process.spclusmultprodxyontrack.wantedSubDets
process.pixeloccupancyxyplotsontrack.multiplicityMaps = cms.VInputTag(cms.InputTag("spclusmultprodxyontrack"))
process.pixeloccupancyxyplotsontrack.occupancyMaps = cms.VInputTag(cms.InputTag("spclusoccuprodxyontrack"))

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

process.seqAnalyzers = cms.Sequence(
    process.goodVertices + process.primaryvertexanalyzer +
    process.pixeloccupancyplots + process.pixeloccupancyplotsontrack +
    process.pixeloccupancyxyplots + process.pixeloccupancyxyplotsontrack)

#-------------------------------------------------------------------------------------------

process.load("Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi")
process.AlignmentTrackSelector.etaMin = cms.double(-5.)
process.AlignmentTrackSelector.etaMax = cms.double(5.)

process.seqProducers = cms.Sequence(process.AlignmentTrackSelector + process.seqMultProd)

process.load("DPGAnalysis.SiStripTools.trackcount_cfi")
process.trackcount.trackCollection = cms.InputTag("generalTracks")
process.trackcount.etaMin= cms.untracked.double(-4.)
process.trackcount.etaMax= cms.untracked.double(4.)
process.trackcount.netabin1D=cms.untracked.uint32(160)
process.trackcount.netabin2D=cms.untracked.uint32(50)
process.trackcount.nchi2bin1D=cms.untracked.uint32(1000)
process.trackcount.nndofbin1D=cms.untracked.uint32(200)
process.trackcount.nchi2bin2D=cms.untracked.uint32(400)
process.trackcount.nndofbin2D=cms.untracked.uint32(100)
process.trackcount.wanted2DHistos=cms.untracked.bool(True)


#----GlobalTag ------------------------

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_condDBv2_cff")
from Configuration.AlCa.GlobalTag_condDBv2 import GlobalTag
#process.GlobalTag = GlobalTag(process.GlobalTag, options.globalTag, '')
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:upgradePLS3', '')

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
                                   fileName = cms.string('OverlapProblem_tpanalyzer_'+options.tag+'.root')
                                   )

process = customise_Reco(process,0)
process = customise_condOverRides(process)

process.myrereco = cms.Sequence(
    process.siPixelRecHits +
    process.trackingGlobalReco)

process.p0 = cms.Path(   process.myrereco +
                         process.seqTrackRefitting
                       + process.trackAssociatorByHits
                       + process.overlapproblemtsosanalyzer
                       + process.overlapproblemtsosall
                       + process.seqProducers
                       + process.seqAnalyzers
                       + process.trackcount
                       )

#print process.dumpPython()
