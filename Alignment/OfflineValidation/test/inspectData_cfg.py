import glob
import FWCore.ParameterSet.Config as cms

###################################################################
# Setup 'standard' options
###################################################################
import FWCore.ParameterSet.VarParsing as VarParsing
options = VarParsing.VarParsing()
options.register('outFileName',
                 "test.root", # default value
                 VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                 VarParsing.VarParsing.varType.string, # string, int, or float
                 "name of the output file (test.root is default)")

options.register('trackCollection',
                 "ctfWithMaterialTracksP5", #ALCARECOTkAlCosmicsCTF0T
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

options.register('inputData',
                 "/eos/cms/store/express/Commissioning2022/ExpressCosmics/FEVT/Express-v1/000/350/010/00000/*",
                 VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                 VarParsing.VarParsing.varType.string, # string, int, or float
                 "eos directory to read from")

options.register('maxEvents',
                 -1,
                 VarParsing.VarParsing.multiplicity.singleton, # singleton or list                 
                 VarParsing.VarParsing.varType.int, # string, int, or float
                 "num. events to run")

options.parseArguments()

process = cms.Process("AlCaRECOAnalysis")

###################################################################
# Message logger service
###################################################################
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.enable = False
process.MessageLogger.DMRChecker=dict()  
process.MessageLogger.GeneralPurposeTrackAnalyzer=dict()
process.MessageLogger.cout = cms.untracked.PSet(
    enable = cms.untracked.bool(True),
    threshold = cms.untracked.string("INFO"),
    default   = cms.untracked.PSet(limit = cms.untracked.int32(0)),                       
    FwkReport = cms.untracked.PSet(limit = cms.untracked.int32(-1),
                                   reportEvery = cms.untracked.int32(1000)
                                   ),                                                      
    DMRChecker = cms.untracked.PSet( limit = cms.untracked.int32(-1)),
    GeneralPurposeTrackAnalyzer = cms.untracked.PSet( limit = cms.untracked.int32(-1)),
    #enableStatistics = cms.untracked.bool(True)
    )

###################################################################
# Geometry producer and standard includes
###################################################################
process.load("RecoVertex.BeamSpotProducer.BeamSpot_cff")
process.load("Configuration.StandardSequences.Services_cff")
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load('Configuration.StandardSequences.MagneticField_cff')
#process.load("Configuration.StandardSequences.MagneticField_0T_cff")
process.load("CondCore.CondDB.CondDB_cfi")

####################################################################
# Get the GlogalTag
####################################################################
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag,options.globalTag, '')

###################################################################
# Source
###################################################################
readFiles = cms.untracked.vstring()
process.source = cms.Source("PoolSource",fileNames = readFiles)
the_files=[]
if(options.unitTest):
    ## fixed input for the unit test
    readFiles.extend(["/store/express/Commissioning2022/ExpressCosmics/FEVT/Express-v1/000/350/010/00000/e0edb947-f8c4-4e6a-b856-ab64117fc6ee.root"]) 
else:
    file_list = glob.glob(options.inputData)
    for f in file_list:
        the_files.append(f.replace("/eos/cms",""))
    print(the_files)
    readFiles.extend(the_files)

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(options.maxEvents))

###################################################################
# momentum constraint for 0T
###################################################################
process.load("RecoTracker.TrackProducer.MomentumConstraintProducer_cff")
import RecoTracker.TrackProducer.MomentumConstraintProducer_cff
process.AliMomConstraint = RecoTracker.TrackProducer.MomentumConstraintProducer_cff.MyMomConstraint.clone()
process.AliMomConstraint.src = options.trackCollection
process.AliMomConstraint.fixedMomentum = 5.0
process.AliMomConstraint.fixedMomentumError = 0.005

###################################################################
# Alignment Track Selector
###################################################################
import Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi
process.MuSkimSelector = Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi.AlignmentTrackSelector.clone(
    applyBasicCuts = True,                                                                            
    filter = True,
    src = options.trackCollection,
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
# The TrackRefitter
###################################################################
process.load("RecoTracker.TrackProducer.TrackRefitters_cff")
import RecoTracker.TrackProducer.TrackRefitters_cff
process.TrackRefitter1 = process.TrackRefitterP5.clone(
    src =  options.trackCollection, #'AliMomConstraint',
    TrajectoryInEvent = True,
    TTRHBuilder = "WithAngleAndTemplate", #"WithTrackAngle"
    NavigationSchool = "",
    #constraint = 'momentum', ### SPECIFIC FOR CRUZET
    #srcConstr='AliMomConstraint' ### SPECIFIC FOR CRUZET$works only with tag V02-10-02 TrackingTools/PatternTools / or CMSSW >=31X
    )

###################################################################
# the pT filter
###################################################################
from CommonTools.RecoAlgos.ptMaxTrackCountFilter_cfi import ptMaxTrackCountFilter
process.myfilter = ptMaxTrackCountFilter.clone(src = cms.InputTag(options.trackCollection),
                                               ptMax = cms.double(3.))

process.preAnaSeq = cms.Sequence()
if(options.unitTest):
    print("adding the max pT filter")
    process.preAnaSeq = cms.Sequence(process.myfilter)

###################################################################
# The analysis module
###################################################################
process.myanalysis = cms.EDAnalyzer("GeneralPurposeTrackAnalyzer",
                                    TkTag  = cms.InputTag('TrackRefitter1'),
                                    isCosmics = cms.bool(True))

process.fastdmr = cms.EDAnalyzer("DMRChecker",
                                 TkTag  = cms.InputTag('TrackRefitter1'),
                                 isCosmics = cms.bool(True))

###################################################################
# Output name
###################################################################
process.TFileService = cms.Service("TFileService",
                                   fileName = cms.string(options.outFileName))

###################################################################
# Path
###################################################################
process.p1 = cms.Path(process.offlineBeamSpot
                      #*process.AliMomConstraint  # for 0T
                      *process.TrackRefitter1
                      *process.myanalysis
                      *process.fastdmr)

###################################################################
# preprend the filter
###################################################################
if(options.unitTest):
    process.p1.insert(0, process.preAnaSeq)
