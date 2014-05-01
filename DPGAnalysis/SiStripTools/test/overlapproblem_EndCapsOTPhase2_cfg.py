import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing

process = cms.Process("OverlapProblem")

#prepare options

options = VarParsing.VarParsing("analysis")

options.register('globalTag',
                 "DONOTEXIST::All",
                 VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                 VarParsing.VarParsing.varType.string,          # string, int, or float
                 "GlobalTag")

options.parseArguments()

#
process.load("DPGAnalysis.SiStripTools.processOptions_cff")
process.load("DPGAnalysis.SiStripTools.MessageLogger_cff")

process.MessageLogger.categories.extend(cms.vstring("NoCluster","ClusterFound"))
process.MessageLogger.cout.placeholder = cms.untracked.bool(False)
process.MessageLogger.cout.threshold = cms.untracked.string("INFO")
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
process.MessageLogger.cout.ClusterFound = cms.untracked.PSet(
    limit = cms.untracked.int32(100000)
    )



process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(options.inputFiles),
                            #                    skipBadFiles = cms.untracked.bool(True),
                            inputCommands = cms.untracked.vstring("keep *", "drop *_MEtoEDMConverter_*_*")
                            )

#process.load("Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff")
#process.load("Configuration.StandardSequences.GeometryDB_cff")
process.load('Configuration.Geometry.GeometryExtendedPhase2TkBE5DPixel10DReco_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_PostLS1_cff')
#process.load("Configuration.Geometry.GeometryExtendedPhaseIPixelReco_cff")
#process.load("Configuration.Geometry.GeometryExtendedPhaseIPixel_cff")
process.load("SimTracker.TrackAssociation.TrackAssociatorByHits_cfi")


from SLHCUpgradeSimulations.Configuration.phase2TkCustomsBE5DPixel10D import customise

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
#process.overlapproblemtsosanalyzer.tsosHMConf.wanted2DHistos = cms.untracked.bool(True)
process.overlapproblemtsosanalyzer.tsosHMConf.wantedSubDets = cms.VPSet(
    cms.PSet(title=cms.string("title"),name = cms.string("TECmD1r1m1"),selection=cms.vstring("0x1fbff004-0x14ac1004")),
    cms.PSet(title=cms.string("title"),name = cms.string("TECmD1r2m1"),selection=cms.vstring("0x1fbff004-0x14ac2004")),
    cms.PSet(title=cms.string("title"),name = cms.string("TECmD1r3m1"),selection=cms.vstring("0x1fbff004-0x14ac3004")),
    cms.PSet(title=cms.string("title"),name = cms.string("TECmD1r4m1"),selection=cms.vstring("0x1fbff004-0x14ac4004")),
    cms.PSet(title=cms.string("title"),name = cms.string("TECmD1r5m1"),selection=cms.vstring("0x1fbff004-0x14ac5004")),
    cms.PSet(title=cms.string("title"),name = cms.string("TECmD1r6m1"),selection=cms.vstring("0x1fbff004-0x14ac6004")),
    cms.PSet(title=cms.string("title"),name = cms.string("TECmD1r7m1"),selection=cms.vstring("0x1fbff004-0x14ac7004")),
    cms.PSet(title=cms.string("title"),name = cms.string("TECmD1r8m1"),selection=cms.vstring("0x1fbff004-0x14ac8004")),
    cms.PSet(title=cms.string("title"),name = cms.string("TECmD1r9m1"),selection=cms.vstring("0x1fbff004-0x14ac9004")),
    cms.PSet(title=cms.string("title"),name = cms.string("TECmD1r10m1"),selection=cms.vstring("0x1fbff004-0x14aca004")),
    cms.PSet(title=cms.string("title"),name = cms.string("TECmD1r11m1"),selection=cms.vstring("0x1fbff004-0x14acb004")),
    cms.PSet(title=cms.string("title"),name = cms.string("TECmD1r12m1"),selection=cms.vstring("0x1fbff004-0x14acc004")),
    cms.PSet(title=cms.string("title"),name = cms.string("TECmD1r13m1"),selection=cms.vstring("0x1fbff004-0x14acd004")),
    cms.PSet(title=cms.string("title"),name = cms.string("TECmD1r14m1"),selection=cms.vstring("0x1fbff004-0x14ace004")),
    cms.PSet(title=cms.string("title"),name = cms.string("TECmD1r15m1"),selection=cms.vstring("0x1fbff004-0x14acf004")),
#
    cms.PSet(title=cms.string("title"),name = cms.string("TECmD1r1m2"),selection=cms.vstring("0x1fbff004-0x14ac1000")),
    cms.PSet(title=cms.string("title"),name = cms.string("TECmD1r2m2"),selection=cms.vstring("0x1fbff004-0x14ac2000")),
    cms.PSet(title=cms.string("title"),name = cms.string("TECmD1r3m2"),selection=cms.vstring("0x1fbff004-0x14ac3000")),
    cms.PSet(title=cms.string("title"),name = cms.string("TECmD1r4m2"),selection=cms.vstring("0x1fbff004-0x14ac4000")),
    cms.PSet(title=cms.string("title"),name = cms.string("TECmD1r5m2"),selection=cms.vstring("0x1fbff004-0x14ac5000")),
    cms.PSet(title=cms.string("title"),name = cms.string("TECmD1r6m2"),selection=cms.vstring("0x1fbff004-0x14ac6000")),
    cms.PSet(title=cms.string("title"),name = cms.string("TECmD1r7m2"),selection=cms.vstring("0x1fbff004-0x14ac7000")),
    cms.PSet(title=cms.string("title"),name = cms.string("TECmD1r8m2"),selection=cms.vstring("0x1fbff004-0x14ac8000")),
    cms.PSet(title=cms.string("title"),name = cms.string("TECmD1r9m2"),selection=cms.vstring("0x1fbff004-0x14ac9000")),
    cms.PSet(title=cms.string("title"),name = cms.string("TECmD1r10m2"),selection=cms.vstring("0x1fbff004-0x14aca000")),
    cms.PSet(title=cms.string("title"),name = cms.string("TECmD1r11m2"),selection=cms.vstring("0x1fbff004-0x14acb000")),
    cms.PSet(title=cms.string("title"),name = cms.string("TECmD1r12m2"),selection=cms.vstring("0x1fbff004-0x14acc000")),
    cms.PSet(title=cms.string("title"),name = cms.string("TECmD1r13m2"),selection=cms.vstring("0x1fbff004-0x14acd000")),
    cms.PSet(title=cms.string("title"),name = cms.string("TECmD1r14m2"),selection=cms.vstring("0x1fbff004-0x14ace000")),
    cms.PSet(title=cms.string("title"),name = cms.string("TECmD1r15m2"),selection=cms.vstring("0x1fbff004-0x14acf000")),
#
    cms.PSet(title=cms.string("title"),name = cms.string("TECxD1r1m1"),selection=cms.vstring("0x1e3ff004-0x142c1004")),
    cms.PSet(title=cms.string("title"),name = cms.string("TECxD1r2m1"),selection=cms.vstring("0x1e3ff004-0x142c2004")),
    cms.PSet(title=cms.string("title"),name = cms.string("TECxD1r3m1"),selection=cms.vstring("0x1e3ff004-0x142c3004")),
    cms.PSet(title=cms.string("title"),name = cms.string("TECxD1r4m1"),selection=cms.vstring("0x1e3ff004-0x142c4004")),
    cms.PSet(title=cms.string("title"),name = cms.string("TECxD1r5m1"),selection=cms.vstring("0x1e3ff004-0x142c5004")),
    cms.PSet(title=cms.string("title"),name = cms.string("TECxD1r6m1"),selection=cms.vstring("0x1e3ff004-0x142c6004")),
    cms.PSet(title=cms.string("title"),name = cms.string("TECxD1r7m1"),selection=cms.vstring("0x1e3ff004-0x142c7004")),
    cms.PSet(title=cms.string("title"),name = cms.string("TECxD1r8m1"),selection=cms.vstring("0x1e3ff004-0x142c8004")),
    cms.PSet(title=cms.string("title"),name = cms.string("TECxD1r9m1"),selection=cms.vstring("0x1e3ff004-0x142c9004")),
    cms.PSet(title=cms.string("title"),name = cms.string("TECxD1r10m1"),selection=cms.vstring("0x1e3ff004-0x142ca004")),
    cms.PSet(title=cms.string("title"),name = cms.string("TECxD1r11m1"),selection=cms.vstring("0x1e3ff004-0x142cb004")),
    cms.PSet(title=cms.string("title"),name = cms.string("TECxD1r12m1"),selection=cms.vstring("0x1e3ff004-0x142cc004")),
    cms.PSet(title=cms.string("title"),name = cms.string("TECxD1r13m1"),selection=cms.vstring("0x1e3ff004-0x142cd004")),
    cms.PSet(title=cms.string("title"),name = cms.string("TECxD1r14m1"),selection=cms.vstring("0x1e3ff004-0x142ce004")),
    cms.PSet(title=cms.string("title"),name = cms.string("TECxD1r15m1"),selection=cms.vstring("0x1e3ff004-0x142cf004")),
#
    cms.PSet(title=cms.string("title"),name = cms.string("TECxD1r1m2"),selection=cms.vstring("0x1e3ff004-0x142c1000")),
    cms.PSet(title=cms.string("title"),name = cms.string("TECxD1r2m2"),selection=cms.vstring("0x1e3ff004-0x142c2000")),
    cms.PSet(title=cms.string("title"),name = cms.string("TECxD1r3m2"),selection=cms.vstring("0x1e3ff004-0x142c3000")),
    cms.PSet(title=cms.string("title"),name = cms.string("TECxD1r4m2"),selection=cms.vstring("0x1e3ff004-0x142c4000")),
    cms.PSet(title=cms.string("title"),name = cms.string("TECxD1r5m2"),selection=cms.vstring("0x1e3ff004-0x142c5000")),
    cms.PSet(title=cms.string("title"),name = cms.string("TECxD1r6m2"),selection=cms.vstring("0x1e3ff004-0x142c6000")),
    cms.PSet(title=cms.string("title"),name = cms.string("TECxD1r7m2"),selection=cms.vstring("0x1e3ff004-0x142c7000")),
    cms.PSet(title=cms.string("title"),name = cms.string("TECxD1r8m2"),selection=cms.vstring("0x1e3ff004-0x142c8000")),
    cms.PSet(title=cms.string("title"),name = cms.string("TECxD1r9m2"),selection=cms.vstring("0x1e3ff004-0x142c9000")),
    cms.PSet(title=cms.string("title"),name = cms.string("TECxD1r10m2"),selection=cms.vstring("0x1e3ff004-0x142ca000")),
    cms.PSet(title=cms.string("title"),name = cms.string("TECxD1r11m2"),selection=cms.vstring("0x1e3ff004-0x142cb000")),
    cms.PSet(title=cms.string("title"),name = cms.string("TECxD1r12m2"),selection=cms.vstring("0x1e3ff004-0x142cc000")),
    cms.PSet(title=cms.string("title"),name = cms.string("TECxD1r13m2"),selection=cms.vstring("0x1e3ff004-0x142cd000")),
    cms.PSet(title=cms.string("title"),name = cms.string("TECxD1r14m2"),selection=cms.vstring("0x1e3ff004-0x142ce000")),
    cms.PSet(title=cms.string("title"),name = cms.string("TECxD1r15m2"),selection=cms.vstring("0x1e3ff004-0x142cf000")),
#
    cms.PSet(title=cms.string("title"),name = cms.string("TECxDxr1m1"),selection=cms.vstring("0x1e33f004-0x14301004")),
    cms.PSet(title=cms.string("title"),name = cms.string("TECxDxr2m1"),selection=cms.vstring("0x1e33f004-0x14302004")),
    cms.PSet(title=cms.string("title"),name = cms.string("TECxDxr3m1"),selection=cms.vstring("0x1e33f004-0x14303004")),
    cms.PSet(title=cms.string("title"),name = cms.string("TECxDxr4m1"),selection=cms.vstring("0x1e33f004-0x14304004")),
    cms.PSet(title=cms.string("title"),name = cms.string("TECxDxr5m1"),selection=cms.vstring("0x1e33f004-0x14305004")),
    cms.PSet(title=cms.string("title"),name = cms.string("TECxDxr6m1"),selection=cms.vstring("0x1e33f004-0x14306004")),
    cms.PSet(title=cms.string("title"),name = cms.string("TECxDxr7m1"),selection=cms.vstring("0x1e33f004-0x14307004")),
    cms.PSet(title=cms.string("title"),name = cms.string("TECxDxr8m1"),selection=cms.vstring("0x1e33f004-0x14308004")),
    cms.PSet(title=cms.string("title"),name = cms.string("TECxDxr9m1"),selection=cms.vstring("0x1e33f004-0x14309004")),
    cms.PSet(title=cms.string("title"),name = cms.string("TECxDxr10m1"),selection=cms.vstring("0x1e33f004-0x1430a004")),
    cms.PSet(title=cms.string("title"),name = cms.string("TECxDxr11m1"),selection=cms.vstring("0x1e33f004-0x1430b004")),
    cms.PSet(title=cms.string("title"),name = cms.string("TECxDxr12m1"),selection=cms.vstring("0x1e33f004-0x1430c004")),
    cms.PSet(title=cms.string("title"),name = cms.string("TECxDxr13m1"),selection=cms.vstring("0x1e33f004-0x1430d004")),
    cms.PSet(title=cms.string("title"),name = cms.string("TECxDxr14m1"),selection=cms.vstring("0x1e33f004-0x1430e004")),
    cms.PSet(title=cms.string("title"),name = cms.string("TECxDxr15m1"),selection=cms.vstring("0x1e33f004-0x1430f004")),
#
    cms.PSet(title=cms.string("title"),name = cms.string("TECxDxr1m2"),selection=cms.vstring("0x1e33f004-0x14301000")),
    cms.PSet(title=cms.string("title"),name = cms.string("TECxDxr2m2"),selection=cms.vstring("0x1e33f004-0x14302000")),
    cms.PSet(title=cms.string("title"),name = cms.string("TECxDxr3m2"),selection=cms.vstring("0x1e33f004-0x14303000")),
    cms.PSet(title=cms.string("title"),name = cms.string("TECxDxr4m2"),selection=cms.vstring("0x1e33f004-0x14304000")),
    cms.PSet(title=cms.string("title"),name = cms.string("TECxDxr5m2"),selection=cms.vstring("0x1e33f004-0x14305000")),
    cms.PSet(title=cms.string("title"),name = cms.string("TECxDxr6m2"),selection=cms.vstring("0x1e33f004-0x14306000")),
    cms.PSet(title=cms.string("title"),name = cms.string("TECxDxr7m2"),selection=cms.vstring("0x1e33f004-0x14307000")),
    cms.PSet(title=cms.string("title"),name = cms.string("TECxDxr8m2"),selection=cms.vstring("0x1e33f004-0x14308000")),
    cms.PSet(title=cms.string("title"),name = cms.string("TECxDxr9m2"),selection=cms.vstring("0x1e33f004-0x14309000")),
    cms.PSet(title=cms.string("title"),name = cms.string("TECxDxr10m2"),selection=cms.vstring("0x1e33f004-0x1430a000")),
    cms.PSet(title=cms.string("title"),name = cms.string("TECxDxr11m2"),selection=cms.vstring("0x1e33f004-0x1430b000")),
    cms.PSet(title=cms.string("title"),name = cms.string("TECxDxr12m2"),selection=cms.vstring("0x1e33f004-0x1430c000")),
    cms.PSet(title=cms.string("title"),name = cms.string("TECxDxr13m2"),selection=cms.vstring("0x1e33f004-0x1430d000")),
    cms.PSet(title=cms.string("title"),name = cms.string("TECxDxr14m2"),selection=cms.vstring("0x1e33f004-0x1430e000")),
    cms.PSet(title=cms.string("title"),name = cms.string("TECxDxr15m2"),selection=cms.vstring("0x1e33f004-0x1430f000")),
    )

process.overlapproblemtsosall = process.overlapproblemtsosanalyzer.clone(onlyValidRecHit = cms.bool(False))

#process.overlapproblemtsosanalyzer.debugMode = cms.untracked.bool(True)
    
from DPGAnalysis.SiStripTools.occupancyplotsselections_phase2_cff import *

process.spclusmultprod = cms.EDProducer("SiPixelClusterMultiplicityProducer",
                                        clusterdigiCollection = cms.InputTag("siPixelClusters"),
                                        wantedSubDets = cms.VPSet()
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

process.seqMultProd = cms.Sequence(
                                   process.spclusmultprod + process.spclusoccuprod +
                                   process.spclusmultprodontrack + process.spclusoccuprodontrack 
                                   )

process.load("DPGAnalysis.SiStripTools.occupancyplots_cfi")
process.occupancyplots.file = cms.untracked.FileInPath("SLHCUpgradeSimulations/Geometry/data/PhaseII/Pixel10D/PixelSkimmedGeometry.txt")

process.pixeloccupancyplots = process.occupancyplots.clone()
process.pixeloccupancyplots.wantedSubDets = cms.VPSet()
process.pixeloccupancyplots.wantedSubDets.extend(OccupancyPlotsPixelWantedSubDets)
process.pixeloccupancyplots.multiplicityMaps = cms.VInputTag(cms.InputTag("spclusmultprod"))
process.pixeloccupancyplots.occupancyMaps = cms.VInputTag(cms.InputTag("spclusoccuprod"))

process.pixeloccupancyplotsontrack = process.occupancyplots.clone()
process.pixeloccupancyplotsontrack.multiplicityMaps = cms.VInputTag(cms.InputTag("spclusmultprodontrack"))
process.pixeloccupancyplotsontrack.occupancyMaps = cms.VInputTag(cms.InputTag("spclusoccuprodontrack"))

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
    process.pixeloccupancyplots + process.pixeloccupancyplotsontrack)

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

process.p0 = cms.Path( process.seqTrackRefitting
                       + process.overlapproblemtsosanalyzer
                       + process.overlapproblemtsosall
                       + process.seqProducers
                       + process.seqAnalyzers
                       + process.trackcount
                       )

#----GlobalTag ------------------------

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
#process.GlobalTag.globaltag = options.globalTag
from Configuration.AlCa.GlobalTag import GlobalTag
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
                                   fileName = cms.string('OverlapProblem_tpanalyzer.root')
                                   )

process = customise(process)

#print process.dumpPython()
