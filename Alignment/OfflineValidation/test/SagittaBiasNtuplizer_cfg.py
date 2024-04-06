import glob
import math
import FWCore.ParameterSet.Config as cms
import FWCore.Utilities.FileUtils as FileUtils
from FWCore.ParameterSet.VarParsing import VarParsing
from Alignment.OfflineValidation.TkAlAllInOneTool.defaultInputFiles_cff import filesDefaultMC_DoubleMuon_string

options = VarParsing('analysis')
options.register('scenario', 
                 'null',
                 VarParsing.multiplicity.singleton,
                 VarParsing.varType.string,
                 "Name of input misalignment scenario")

options.register('globalTag',
                 "'auto:phase1_2022_realistic", # default value
                 VarParsing.multiplicity.singleton, # singleton or list
                 VarParsing.varType.string, # string, int, or float
                 "name of the input Global Tag")

options.register ('myfile',
                  filesDefaultMC_DoubleMuon_string, # default value
                  VarParsing.multiplicity.singleton,
                  VarParsing.varType.string,
                  "file name")

options.register ('FileList',
                  '', # default value
                  VarParsing.multiplicity.singleton, 
                  VarParsing.varType.string,
                  "FileList in DAS format")

options.parseArguments()

if(options.FileList):
    print("FileList:           ", options.FileList)
else:
    print("inputFile:          ", options.myfile)
print("outputFile:         ", "ZmmNtuple_MC_GEN-SIM_{fscenario}.root".format(fscenario=options.scenario))
print("conditionGT:        ", options.globalTag)
print("max events:         ", options.maxEvents)

valid_scenarios = ['-10e-6','-8e-6','-6e-6','-4e-6','-2e-6','0','2e-6','4e-6','6e-6','8e-6','10e-6','null']

if options.scenario not in valid_scenarios:
    print("Error: Invalid scenario specified. Please choose from the following list: ")
    print(valid_scenarios)
    exit(1)

process = cms.Process("SagittaBiasNtuplizer")

###################################################################
# Set the process to run multi-threaded
###################################################################
process.options.numberOfThreads = 8

###################################################################
# Message logger service
###################################################################
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.enable = False
process.MessageLogger.SagittaBiasNtuplizer=dict()  
process.MessageLogger.cout = cms.untracked.PSet(
    enable = cms.untracked.bool(True),
    threshold = cms.untracked.string("INFO"),
    default   = cms.untracked.PSet(limit = cms.untracked.int32(0)),                       
    FwkReport = cms.untracked.PSet(limit = cms.untracked.int32(-1),
                                   reportEvery = cms.untracked.int32(1000)
                                   ),                                                      
    SagittaBiasNtuplizer = cms.untracked.PSet( limit = cms.untracked.int32(-1)),
    enableStatistics = cms.untracked.bool(True)
    )

###################################################################
# Geometry producer and standard includes
###################################################################
process.load("RecoVertex.BeamSpotProducer.BeamSpot_cff")
process.load("Configuration.StandardSequences.Services_cff")
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load("CondCore.CondDB.CondDB_cfi")

###################################################################
# TransientTrack from https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideTransientTracks
###################################################################
process.load("TrackingTools.TransientTrack.TransientTrackBuilder_cfi")
process.load('TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorOpposite_cfi')
process.load('TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAlong_cfi')
process.load('TrackingTools.TrackAssociator.DetIdAssociatorESProducer_cff')

####################################################################
# Get the GlogalTag
####################################################################
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag,"125X_mcRun3_2022_design_v6", '')
if (options.scenario=='null'):
    print("null scenario, do nothing")
    pass
elif (options.scenario=='ideal'):
    print("ideal scenario, use ideal tags")
    process.GlobalTag.toGet = cms.VPSet(cms.PSet(record = cms.string('TrackerAlignmentRcd'),
                                                 tag = cms.string("TrackerAlignment_Upgrade2017_design_v4")),
                                        cms.PSet(record = cms.string('TrackerAlignmentErrorExtendedRcd'),
                                                 tag = cms.string("TrackerAlignmentErrorsExtended_Upgrade2017_design_v0")),
                                        cms.PSet(record = cms.string('TrackerSurfaceDeformationRcd'),
                                                 tag = cms.string("TrackerSurfaceDeformations_zero")))
else :
    print("using {} scenario".format(options.scenario))
    process.GlobalTag.toGet = cms.VPSet(cms.PSet(connect = cms.string("sqlite_file:/afs/cern.ch/user/m/musich/public/layer_rotation_studies/outputfile_"+options.scenario+".db"),                                             
                                                 record = cms.string('TrackerAlignmentRcd'),
                                                 tag = cms.string("Alignments")))

###################################################################
# Source
###################################################################
if(options.FileList):
    print('Loading file list from ASCII file')
    filelist = FileUtils.loadListFromFile (options.FileList)
    readFiles = cms.untracked.vstring( *filelist)
else:
    readFiles = cms.untracked.vstring([options.myfile])

process.source = cms.Source("PoolSource",
                            fileNames = readFiles)

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(options.maxEvents))

###################################################################
# Alignment Track Selector
###################################################################
import Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi
process.MuSkimSelector = Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi.AlignmentTrackSelector.clone(
    applyBasicCuts = True,                                                                            
    filter = True,
    src = "ALCARECOTkAlDiMuon",
    ptMin = 17.,
    pMin = 17.,
    etaMin = -2.5,
    etaMax = 2.5,
    d0Min = -2.,
    d0Max = 2.,
    dzMin = -25.,
    dzMax = 25.,
    nHitMin = 6,
    nHitMin2D = 0)

###################################################################
# refitting the muon tracks
###################################################################
process.load("RecoTracker.TrackProducer.TrackRefitters_cff")
import RecoTracker.TrackProducer.TrackRefitters_cff
process.refittedMuons = RecoTracker.TrackProducer.TrackRefitter_cfi.TrackRefitter.clone(
    src = "ALCARECOTkAlDiMuon",
    TrajectoryInEvent = True,
    NavigationSchool = '',
    TTRHBuilder = "WithAngleAndTemplate")

###################################################################
# refitting the vertex tracks
###################################################################
process.refittedVtxTracks = RecoTracker.TrackProducer.TrackRefitter_cfi.TrackRefitter.clone(
    src = "ALCARECOTkAlDiMuonVertexTracks",
    TrajectoryInEvent = True,
    NavigationSchool = '',
    TTRHBuilder = "WithAngleAndTemplate")

###################################################################
# refitting all tracks
###################################################################
process.refittedTracks = RecoTracker.TrackProducer.TrackRefitter_cfi.TrackRefitter.clone(
    src = "generalTracks",
    TrajectoryInEvent = True,
    NavigationSchool = '',
    TTRHBuilder = "WithAngleAndTemplate")

####################################################################
# Re-do vertices
####################################################################
from RecoVertex.PrimaryVertexProducer.OfflinePrimaryVertices_cfi import offlinePrimaryVertices
process.offlinePrimaryVerticesFromRefittedTrks = offlinePrimaryVertices.clone()
#process.offlinePrimaryVerticesFromRefittedTrks.TrackLabel = cms.InputTag("refittedVtxTracks")
process.offlinePrimaryVerticesFromRefittedTrks.TrackLabel = cms.InputTag("refittedTracks")

###################################################################
# The analysis modules
###################################################################
process.ZtoMMNtuple = cms.EDAnalyzer("SagittaBiasNtuplizer",
                                     #tracks = cms.InputTag('refittedMuons'),
                                     useReco = cms.bool(True),
                                     muons = cms.InputTag('muons'),
                                     doGen = cms.bool(True),
                                     tracks = cms.InputTag('refittedTracks'),
                                     vertices = cms.InputTag('offlinePrimaryVerticesFromRefittedTrks'))

process.DiMuonVertexValidation = cms.EDAnalyzer("DiMuonVertexValidation",
                                                useReco = cms.bool(False),
                                                ## the two parameters below are mutually exclusive,
                                                ## depending if RECO or ALCARECO is used
                                                #muons  = cms.InputTag(''),
                                                muonTracks = cms.InputTag('refittedMuons'),
                                                tracks = cms.InputTag(''),
                                                vertices = cms.InputTag('offlinePrimaryVerticesFromRefittedTrks'))
                                    
from Alignment.OfflineValidation.diMuonValidation_cfi import diMuonValidation as _diMuonValidation
process.DiMuonMassValidation = _diMuonValidation.clone(
    TkTag = 'refittedMuons',
    #TkTag = 'TrackRefitter1',
    # mu mu mass
    Pair_mass_min   = 80.,
    Pair_mass_max   = 120.,
    Pair_mass_nbins = 80,
    Pair_etaminpos  = -2.4,
    Pair_etamaxpos  = 2.4,
    Pair_etaminneg  = -2.4,
    Pair_etamaxneg  = 2.4,
    # cosTheta CS
    Variable_CosThetaCS_xmin  = -1.,
    Variable_CosThetaCS_xmax  =  1.,
    Variable_CosThetaCS_nbins = 20,
    # DeltaEta
    Variable_DeltaEta_xmin  = -4.8,
    Variable_DeltaEta_xmax  = 4.8,
    Variable_DeltaEta_nbins = 20,
    # EtaMinus
    Variable_EtaMinus_xmin  = -2.4,
    Variable_EtaMinus_xmax  =  2.4,
    Variable_EtaMinus_nbins = 12,
    # EtaPlus
    Variable_EtaPlus_xmin  = -2.4,
    Variable_EtaPlus_xmax  =  2.4,
    Variable_EtaPlus_nbins = 12,
    # Phi CS
    Variable_PhiCS_xmin  = -math.pi/2.,
    Variable_PhiCS_xmax  =  math.pi/2.,
    Variable_PhiCS_nbins = 20,
    # Phi Minus
    Variable_PhiMinus_xmin  = -math.pi,
    Variable_PhiMinus_xmax  =  math.pi,
    Variable_PhiMinus_nbins = 16,
    # Phi Plus
    Variable_PhiPlus_xmin  = -math.pi,
    Variable_PhiPlus_xmax  =  math.pi,
    Variable_PhiPlus_nbins = 16,
    # mu mu pT
    Variable_PairPt_xmin  = 0.,
    Variable_PairPt_xmax  = 100.,
    Variable_PairPt_nbins = 100)

###################################################################
# Output name
###################################################################
process.TFileService = cms.Service("TFileService",
                                   fileName = cms.string("ZmmNtuple_MC_GEN-SIM_"+options.scenario+".root"))

###################################################################
# Path
###################################################################
process.p1 = cms.Path(process.offlineBeamSpot
                      #* process.refittedMuons
                      #* process.refittedVtxTracks
                      * process.refittedTracks
                      * process.offlinePrimaryVerticesFromRefittedTrks
                      * process.ZtoMMNtuple) 
                      #* process.DiMuonVertexValidation
                      #* process.DiMuonMassValidation)
