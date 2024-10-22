import FWCore.ParameterSet.Config as cms
from Alignment.OfflineValidation.TkAlAllInOneTool.defaultInputFiles_cff import filesDefaultData_Comissioning2022_Cosmics_string

###################################################################
# Setup 'standard' options
###################################################################
import FWCore.ParameterSet.VarParsing as VarParsing
options = VarParsing.VarParsing()
options.register('outFileName',
                 "CosmicTrackSplitterValidation.root", # default value
                 VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                 VarParsing.VarParsing.varType.string, # string, int, or float
                 "name of the output file (test.root is default)")

options.register('trackCollection',
                 "ALCARECOTkAlCosmicsCTF0T", #"ctfWithMaterialTracksP5"
                 VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                 VarParsing.VarParsing.varType.string, # string, int, or float
                 "name of the input track collection")

options.register('globalTag',
                 "auto:run3_data_prompt", # default value
                 VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                 VarParsing.VarParsing.varType.string, # string, int, or float
                 "name of the input Global Tag")

options.register('unitTest',
                 False, # default value
                 VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                 VarParsing.VarParsing.varType.bool, # string, int, or float
                 "is it a unit test?")

options.register('maxEvents',
                 -1,
                 VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                 VarParsing.VarParsing.varType.int, # string, int, or float
                 "num. events to run")

options.parseArguments()

###################################################################
# process name: should be used in the CosmicSplitterValidation config!
###################################################################
process = cms.Process("splitter")

###################################################################
# message logger
###################################################################
process.load("FWCore.MessageLogger.MessageLogger_cfi")
## report only every 100th record
process.MessageLogger.cerr.FwkReport.reportEvery = 1 if (options.unitTest) else 100

###################################################################
# magnetic field
###################################################################
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load('Configuration.StandardSequences.Services_cff')
process.load("Configuration.Geometry.GeometryRecoDB_cff")

###################################################################
# including global tag
###################################################################
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag,options.globalTag, '')

###################################################################
# track selectors and refitting
###################################################################
process.load("Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi")
process.load("RecoVertex.BeamSpotProducer.BeamSpot_cff")
process.load("RecoTracker.TrackProducer.TrackRefitters_cff")

###################################################################
# event source
###################################################################
readFiles = cms.untracked.vstring()
process.source = cms.Source("PoolSource",fileNames = readFiles)
if(options.unitTest):
    ## fixed input for the unit test
    readFiles.extend([filesDefaultData_Comissioning2022_Cosmics_string])
else :
    readFiles.extend([]) # put here your file list
    print('No input files specified!')

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32((10 if (options.unitTest) else options.maxEvents)))

###################################################################
# adding geometries
###################################################################
from CondCore.CondDB.CondDB_cfi import *
CondDBConnection = CondDB.clone(connect = 'frontier://FrontierProd/CMS_CONDITIONS')

###################################################################
# CRAFT REPRO geom
###################################################################
process.trackerAlignment = cms.ESSource("PoolDBESSource",
                                        CondDBConnection,
                                        toGet = cms.VPSet(cms.PSet(
                                            record = cms.string('TrackerAlignmentRcd'),
                                            tag = cms.string('TrackerAlignment_v30_offline')
                                        )))

###################################################################
# APEs REPRO
###################################################################
process.trackerAPE = cms.ESSource("PoolDBESSource",
                                  CondDBConnection,
                                  toGet = cms.VPSet(cms.PSet(
                                      record = cms.string('TrackerAlignmentErrorExtendedRcd'),
                                      tag = cms.string('TrackerAlignmentExtendedErrors_v16_offline_IOVs')
                                  )))

###################################################################
# set prefer
###################################################################
process.es_prefer_trackerAlignment = cms.ESPrefer("PoolDBESSource", "trackerAlignment")
process.es_prefer_trackerAPE = cms.ESPrefer("PoolDBESSource", "trackerAPE")

# # hit filter
# process.load("RecoTracker.FinalTrackSelectors.TrackerTrackHitFilter_cff")
# # parameters for TrackerTrackHitFilter
# #process.TrackerTrackHitFilter.src = "cosmictrackfinderP5"
# #process.TrackerTrackHitFilter.src = 'ALCARECOTkAlCosmicsCTF'
# #process.TrackerTrackHitFilter.src = 'ALCARECOTkAlCosmicsCTF0T'
# process.TrackerTrackHitFilter.src = options.trackCollection
# process.TrackerTrackHitFilter.rejectBadStoNHits = True
# process.TrackerTrackHitFilter.TrackAngleCut = 0.1

# # re-build the track
# import RecoTracker.TrackProducer.CTFFinalFitWithMaterialP5_cff   #TrackRefitters_cff
# process.HitFilteredTracks = RecoTracker.TrackProducer.CTFFinalFitWithMaterialP5_cff.ctfWithMaterialTracksCosmics.clone(
#     src = 'TrackerTrackHitFilter',
#     TTRHBuilder = "WithAngleAndTemplate"
# )

# # refit tracks first
# process.TrackRefitterP5.src = 'HitFilteredTracks'
# #process.TrackRefitterP5.src = "ALCARECOTkAlCosmicsCosmicTF0T"
# process.TrackRefitterP5.TTRHBuilder = "WithTrackAngle"
# process.FittingSmootherRKP5.EstimateCut = -1

# # module configuration
# # alignment track selector
# #process.AlignmentTrackSelector.src = "ALCARECOTkAlCosmicsCTF0T"
# #process.AlignmentTrackSelector.src = "TrackerTrackHitFilter"
# process.AlignmentTrackSelector.src = "TrackRefitterP5"
# process.AlignmentTrackSelector.filter = True
# process.AlignmentTrackSelector.applyBasicCuts = True
# process.AlignmentTrackSelector.ptMin   = 0.
# process.AlignmentTrackSelector.pMin   = 4.
# process.AlignmentTrackSelector.ptMax   = 9999.
# process.AlignmentTrackSelector.pMax   = 9999.
# process.AlignmentTrackSelector.etaMin  = -9999.
# process.AlignmentTrackSelector.etaMax  = 9999.
# process.AlignmentTrackSelector.nHitMin = 10
# process.AlignmentTrackSelector.nHitMin2D = 2
# process.AlignmentTrackSelector.chi2nMax = 9999.
# process.AlignmentTrackSelector.applyMultiplicityFilter = True
# process.AlignmentTrackSelector.maxMultiplicity = 1
# process.AlignmentTrackSelector.applyNHighestPt = False
# process.AlignmentTrackSelector.nHighestPt = 1
# process.AlignmentTrackSelector.seedOnlyFrom = 0
# process.AlignmentTrackSelector.applyIsolationCut = False
# process.AlignmentTrackSelector.minHitIsolation = 0.8
# process.AlignmentTrackSelector.applyChargeCheck = False
# process.AlignmentTrackSelector.minHitChargeStrip = 50.
# process.AlignmentTrackSelector.minHitsPerSubDet.inBPIX = 2
# process.KFFittingSmootherWithOutliersRejectionAndRK.EstimateCut=30.0
# process.KFFittingSmootherWithOutliersRejectionAndRK.MinNumberOfHits=4

# # configuration of the track spitting module
# # new cuts allow for cutting on the impact parameter of the original track
# process.load("RecoTracker.FinalTrackSelectors.cosmicTrackSplitter_cfi")
# process.cosmicTrackSplitter.tracks='AlignmentTrackSelector'
# process.cosmicTrackSplitter.tjTkAssociationMapTag='TrackRefitterP5'

# #---------------------------------------------------------------------
# # the output of the track hit filter are track candidates
# # give them to the TrackProducer
# process.ctfWithMaterialTracksP5.src = 'cosmicTrackSplitter'

# # second refit
# import RecoTracker.TrackProducer.TrackRefitters_cff
# process.TrackRefitter2 = RecoTracker.TrackProducer.TrackRefitterP5_cfi.TrackRefitterP5.clone()
# process.TrackRefitter2.src = 'ctfWithMaterialTracksP5'
# process.TrackRefitter2.TTRHBuilder = "WithTrackAngle"

# process.p = cms.Path(
#     process.offlineBeamSpot *
#     process.TrackerTrackHitFilter *
#     process.HitFilteredTracks *
#     process.TrackRefitterP5 *
#     process.AlignmentTrackSelector *
#     process.cosmicTrackSplitter *
#     process.ctfWithMaterialTracksP5 *
#     process.TrackRefitter2 *
#     process.cosmicValidation)

###################################################################
# common track selection / refitter sequence
###################################################################
import Alignment.CommonAlignment.tools.trackselectionRefitting as trackselRefit
process.seqTrackselRefit = trackselRefit.getSequence(process, options.trackCollection ,
                                                     isPVValidation=False,
                                                     TTRHBuilder='WithAngleAndTemplate',
                                                     usePixelQualityFlag=True,
                                                     openMassWindow=False,
                                                     cosmicsDecoMode=True,
                                                     cosmicsZeroTesla=False,
                                                     momentumConstraint=None,
                                                     cosmicTrackSplitting=True,
                                                     use_d0cut=False)

###################################################################
# adding this ~doubles the efficiency of selection (!)
###################################################################
process.FittingSmootherRKP5.EstimateCut = -1

###################################################################
# the analysis module
###################################################################
from Alignment.OfflineValidation.cosmicSplitterValidation_cfi import cosmicSplitterValidation as _cosmicSplitterValidation
process.cosmicValidation = _cosmicSplitterValidation.clone(
    ifSplitMuons        = False,
    checkIfGolden       = False,
    splitTracks         = ("FinalTrackRefitter","","splitter"),  # important the 3rd argument should be the name of the process!
    splitGlobalMuons    = ("muons","","splitter"),               # important the 3rd argument should be the name of the process!
    originalTracks      = ("FirstTrackRefitter","","splitter"),  # important the 3rd argument should be the name of the process!
    originalGlobalMuons = ("muons","","Rec"))

###################################################################
# Output file
###################################################################
process.TFileService = cms.Service("TFileService",
                                   fileName = cms.string(options.outFileName))

###################################################################
# path
###################################################################
process.p = cms.Path(
    process.seqTrackselRefit *
    process.cosmicValidation)
