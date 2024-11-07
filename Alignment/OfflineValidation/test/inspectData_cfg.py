import math
import glob
import importlib
import FWCore.ParameterSet.Config as cms
from Alignment.OfflineValidation.TkAlAllInOneTool.defaultInputFiles_cff import filesDefaultData_Comissioning2022_Cosmics_string,filesDefaultMC_DoubleMuonPUPhase_string

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

options.register('isDiMuonData',
                 False, # default value
                 VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                 VarParsing.VarParsing.varType.bool, # string, int, or float
                 "is it running on DiMuon data?")

options.register('isCosmics',
                 False, # default value
                 VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                 VarParsing.VarParsing.varType.bool, # string, int, or float
                 "is it running on cosmics data?")

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

options.register('Detector',
                 '2023',
                 VarParsing.VarParsing.multiplicity.singleton, # singleton or list                 
                 VarParsing.VarParsing.varType.string, # string, int, or float
                 "Detector to run upon")

options.parseArguments()


from Configuration.PyReleaseValidation.upgradeWorkflowComponents import upgradeProperties
ConditionsInfo = {}
if 'D' in options.Detector:
    ConditionsInfo = upgradeProperties['Run4'][options.Detector] # so if the default changes, change wf only here
else:
    ConditionsInfo = upgradeProperties[2017][options.Detector]
    
era_value = ConditionsInfo['Era']
era_module_name = f'Configuration.Eras.Era_{era_value}_cff'
config_name = f'{era_value}'
era_module = importlib.import_module(era_module_name)
era_config = getattr(era_module, config_name, None)

if era_config is not None:
    # Use the configurations from the imported module in the process setup
    process = cms.Process("AlCaRECOAnalysis", era_config)
else:
    print(f"Error: Could not find configuration {config_name} in module {era_module_name}.")

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
if 'D' in options.Detector:
    geom = options.Detector  # Replace with your actual dynamic part
    process.load(f'Configuration.Geometry.GeometryExtended{geom}Reco_cff')
else:
    process.load("Configuration.StandardSequences.GeometryRecoDB_cff")

process.load('Configuration.StandardSequences.MagneticField_cff')
#process.load("Configuration.StandardSequences.MagneticField_0T_cff")
process.load("CondCore.CondDB.CondDB_cfi")

####################################################################
# Get the GlogalTag
####################################################################
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag,options.globalTag if (options.globalTag != '') else ConditionsInfo['GT'], '')

###################################################################
# Source
###################################################################
readFiles = cms.untracked.vstring()
process.source = cms.Source("PoolSource",fileNames = readFiles)
the_files=[]
if(options.unitTest):
    ## fixed input for the unit test
    if('D' in options.Detector) :
        # it's for phase-2
        readFiles.extend([filesDefaultMC_DoubleMuonPUPhase_string])
    else:
        # it's for phase-1
        readFiles.extend([filesDefaultData_Comissioning2022_Cosmics_string])
else:
    file_list = glob.glob(options.inputData)
    for f in file_list:
        if '/eos/cms' in f:
            the_files.append(f.replace("/eos/cms",""))
        else: 
            the_files.append(f.replace("./","file:"))
    print(the_files)
    readFiles.extend(the_files)

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32((10 if (options.unitTest) else options.maxEvents)))

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
if options.isCosmics:
    process.TrackRefitter1 = process.TrackRefitterP5.clone(
        src =  options.trackCollection, #'AliMomConstraint',
        TrajectoryInEvent = True,
        TTRHBuilder = "WithAngleAndTemplate", #"WithTrackAngle"
        NavigationSchool = "",
        #constraint = 'momentum', ### SPECIFIC FOR CRUZET
        #srcConstr='AliMomConstraint' ### SPECIFIC FOR CRUZET$works only with tag V02-10-02 TrackingTools/PatternTools / or CMSSW >=31X
    )
else:
    process.TrackRefitter1 = RecoTracker.TrackProducer.TrackRefitter_cfi.TrackRefitter.clone(
        src =  options.trackCollection, #'AliMomConstraint',
        TrajectoryInEvent = True,
        TTRHBuilder = "WithAngleAndTemplate", #"WithTrackAngle"
        NavigationSchool = "",
    )

###################################################################
# the pT filter
###################################################################
from CommonTools.RecoAlgos.ptMaxTrackCountFilter_cfi import ptMaxTrackCountFilter
process.myfilter = ptMaxTrackCountFilter.clone(src = cms.InputTag(options.trackCollection),
                                               ptMax = cms.double(10.))

process.preAnaSeq = cms.Sequence()
if(options.unitTest):
    print("adding the max pT filter")
    process.preAnaSeq = cms.Sequence(process.myfilter)

###################################################################
# The analysis module
###################################################################
process.myanalysis = cms.EDAnalyzer("GeneralPurposeTrackAnalyzer",
                                    TkTag  = cms.InputTag('TrackRefitter1'),
                                    #TkTag  = cms.InputTag(options.trackCollection),
                                    isCosmics = cms.bool(options.isCosmics))

process.fastdmr = cms.EDAnalyzer("DMRChecker",
                                 TkTag  = cms.InputTag('TrackRefitter1'),
                                 isCosmics = cms.bool(options.isCosmics))

###################################################################
# Output name
###################################################################
process.TFileService = cms.Service("TFileService",
                                   fileName = cms.string(options.outFileName))


###################################################################
# TransientTrack from https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideTransientTracks
###################################################################
process.load("TrackingTools.TransientTrack.TransientTrackBuilder_cfi")
process.load('TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorOpposite_cfi')
process.load('TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAlong_cfi')
process.load('TrackingTools.TrackAssociator.DetIdAssociatorESProducer_cff')

process.DiMuonVertexValidation = cms.EDAnalyzer("DiMuonVertexValidation",
                                                useReco = cms.bool(False),
                                                muonTracks = cms.InputTag('TrackRefitter1'),
                                                tracks = cms.InputTag(''),
                                                vertices = cms.InputTag('offlinePrimaryVertices'))

from Alignment.OfflineValidation.diMuonValidation_cfi import diMuonValidation as _diMuonValidation
process.DiMuonMassValidation = _diMuonValidation.clone(
    #TkTag = 'refittedMuons',
    TkTag = 'TrackRefitter1',
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
# Path
###################################################################
process.p1 = cms.Path(process.offlineBeamSpot
                      #*process.AliMomConstraint  # for 0T
                      * process.TrackRefitter1
                      * process.myanalysis
                      * process.fastdmr)

###################################################################
# append di muon analysis
###################################################################
if(options.isDiMuonData):
    process.p1.insert(5,process.DiMuonVertexValidation)
    process.p1.insert(6,process.DiMuonMassValidation)

###################################################################
# preprend the filter for unit tests
###################################################################
if(options.unitTest and not options.isDiMuonData):
    process.p1.insert(0, process.preAnaSeq)



    
