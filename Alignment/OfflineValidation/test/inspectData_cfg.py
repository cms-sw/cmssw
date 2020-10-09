import glob
import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing
options = VarParsing.VarParsing()

###################################################################
# Setup 'standard' options
###################################################################

options.register('OutFileName',
                 "test.root", # default value
                 VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                 VarParsing.VarParsing.varType.string, # string, int, or float
                 "name of the output file (test.root is default)")

options.register('myGT',
                 "auto:phase1_2021_cosmics_0T", # default value
                 VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                 VarParsing.VarParsing.varType.string, # string, int, or float
                 "name of the input Global Tag")

options.register('myDataset',
                 "myDataset_v1",
                 VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                 VarParsing.VarParsing.varType.string, # string, int, or float
                 "name of the input dataset")

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
process.MessageLogger.cerr = cms.untracked.PSet(placeholder = cms.untracked.bool(True))
process.MessageLogger.cout = cms.untracked.PSet(INFO = cms.untracked.PSet(
    reportEvery = cms.untracked.int32(1000) # every 100th only
    #    limit = cms.untracked.int32(10)       # or limit to 10 printouts...
    ))

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
process.GlobalTag = GlobalTag(process.GlobalTag,options.myGT, '')

# process.GlobalTag.toGet = cms.VPSet(
#     cms.PSet(record = cms.string("SiPixelTemplateDBObjectRcd"),
#              label = cms.untracked.string("0T"),
#              #tag = cms.string("SiPixelTemplateDBObject_phase1_0T_mc_BoR3_v1"),
#              tag = cms.string("SiPixelTemplateDBObject_phase1_0T_mc_BoR3_v1_bugfix"),
#              connect = cms.string("frontier://FrontierProd/CMS_CONDITIONS")
#              ),
#     cms.PSet(record = cms.string("SiPixelGenErrorDBObjectRcd"),
#              #tag = cms.string("SiPixelGenErrorDBObject_phase1_0T_mc_BoR3_v1"),
#              tag = cms.string("SiPixelGenErrorDBObject_phase1_0T_mc_BoR3_v1_bugfix"),
#              connect = cms.string("frontier://FrontierProd/CMS_CONDITIONS")
#              ),
# )


###################################################################
# Source
###################################################################

readFiles = cms.untracked.vstring()
process.source = cms.Source("PoolSource",fileNames = readFiles)
the_files=[]
file_list = glob.glob("/eos/cms/store/mc/Run3Winter20CosmicDR/TKCosmics_0T/ALCARECO/TkAlCosmics0T-0T_110X_mcRun3_2021cosmics_realistic_deco_v4-v1/40000/*")
for f in file_list:
    the_files.append(f.replace("/eos/cms",""))

readFiles.extend(the_files)
    
print(the_files)
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(options.maxEvents))

# readFiles.extend(['/store/mc/Run3Winter20CosmicDR/TKCosmics_0T/ALCARECO/TkAlCosmics0T-0T_110X_mcRun3_2021cosmics_realistic_deco_v4-v1/40000/4AC4A0B1-12CD-CC4C-AA44-BF94D02DA323.root',
#                   '/store/mc/Run3Winter20CosmicDR/TKCosmics_0T/ALCARECO/TkAlCosmics0T-0T_110X_mcRun3_2021cosmics_realistic_deco_v4-v1/40000/44D2D421-9B16-FA47-9C53-B8B93A7F2077.root',
#                   '/store/mc/Run3Winter20CosmicDR/TKCosmics_0T/ALCARECO/TkAlCosmics0T-0T_110X_mcRun3_2021cosmics_realistic_deco_v4-v1/40000/525F1D6D-9271-FF41-B05E-C99FFCB029D9.root',
#                   '/store/mc/Run3Winter20CosmicDR/TKCosmics_0T/ALCARECO/TkAlCosmics0T-0T_110X_mcRun3_2021cosmics_realistic_deco_v4-v1/40000/39381C24-97B6-E641-B4AB-7D1B1D25A782.root',
#                   '/store/mc/Run3Winter20CosmicDR/TKCosmics_0T/ALCARECO/TkAlCosmics0T-0T_110X_mcRun3_2021cosmics_realistic_deco_v4-v1/40000/08901961-CFB6-4A43-98FC-D1817EF76D13.root'])


###################################################################
# momentum constraint for 0T
###################################################################
process.load("RecoTracker.TrackProducer.MomentumConstraintProducer_cff")
import RecoTracker.TrackProducer.MomentumConstraintProducer_cff
process.AliMomConstraint = RecoTracker.TrackProducer.MomentumConstraintProducer_cff.MyMomConstraint.clone()
process.AliMomConstraint.src = 'ALCARECOTkAlCosmicsCTF0T'
process.AliMomConstraint.fixedMomentum = 5.0
process.AliMomConstraint.fixedMomentumError = 0.005

###################################################################
# Alignment Track Selector
###################################################################
import Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi

process.MuSkimSelector = Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi.AlignmentTrackSelector.clone(
    applyBasicCuts = True,                                                                            
    filter = True,
    src = 'ALCARECOTkAlCosmicsCTF0T',
    ptMin = 17.,
    pMin = 17.,
    etaMin = -2.5,
    etaMax = 2.5,
    d0Min = -2.,
    d0Max = 2.,
    dzMin = -25.,
    dzMax = 25.,
    nHitMin = 6,
    nHitMin2D = 0,
    )

###################################################################
# The TrackRefitter
###################################################################
process.load("RecoTracker.TrackProducer.TrackRefitters_cff")
import RecoTracker.TrackProducer.TrackRefitters_cff
process.TrackRefitter1 = process.TrackRefitterP5.clone(
    src =  'ALCARECOTkAlCosmicsCTF0T', #'AliMomConstraint',
    TrajectoryInEvent = True,
    TTRHBuilder = "WithTrackAngle",  #"WithAngleAndTemplate",
    NavigationSchool = "",
    #constraint = 'momentum', ### SPECIFIC FOR CRUZET
    #srcConstr='AliMomConstraint' ### SPECIFIC FOR CRUZET$works only with tag V02-10-02 TrackingTools/PatternTools / or CMSSW >=31X
    )

###################################################################
# The analysis module
###################################################################
process.myanalysis = cms.EDAnalyzer("GeneralPurposeTrackAnalyzer",
                                    TkTag  = cms.InputTag('TrackRefitter1'),
                                    isCosmics = cms.bool(True)
                                    )

process.fastdmr = cms.EDAnalyzer("DMRChecker",
                                 TkTag  = cms.InputTag('TrackRefitter1'),
                                 isCosmics = cms.bool(True)
                                 )

###################################################################
# Output name
###################################################################
process.TFileService = cms.Service("TFileService",
                                   fileName = cms.string(options.OutFileName)
                                   )

###################################################################
# Path
###################################################################
process.p1 = cms.Path(process.offlineBeamSpot*
                      #process.AliMomConstraint*
                      process.TrackRefitter1*
                      process.myanalysis*
                      process.fastdmr
                      )
