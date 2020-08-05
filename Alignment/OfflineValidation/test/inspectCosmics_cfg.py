import glob
import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing
options = VarParsing.VarParsing()

###################################################################
# Setup 'standard' options
###################################################################

options.register('OutFileName',
                 "test3.root", # default value
                 VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                 VarParsing.VarParsing.varType.string, # string, int, or float
                 "name of the output file (test.root is default)")

options.register('myGT',
                 "110X_mcRun3_2021cosmics_realistic_deco_v4", # default value
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
#process.load('Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff')
process.load("Configuration.StandardSequences.MagneticField_0T_cff")
process.load("CondCore.CondDB.CondDB_cfi")

###################################################################
# Option 1: just state the Global Tag 
###################################################################
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = options.myGT

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
#process.source = cms.Source ("PoolSource",fileNames =  cms.untracked.vstring())
#process.source.fileNames = [options.InputFileName]

process.source = cms.Source ("PoolSource",fileNames =  cms.untracked.vstring())
process.source.fileNames = ['/store/mc/Run3Winter20CosmicDR/TKCosmics_0T/ALCARECO/TkAlCosmics0T-0T_110X_mcRun3_2021cosmics_realistic_deco_v4-v1/40000/4AC4A0B1-12CD-CC4C-AA44-BF94D02DA323.root',
                            '/store/mc/Run3Winter20CosmicDR/TKCosmics_0T/ALCARECO/TkAlCosmics0T-0T_110X_mcRun3_2021cosmics_realistic_deco_v4-v1/40000/7E1DDB72-0C74-E642-8892-5219274E53B2.root',
                            '/store/mc/Run3Winter20CosmicDR/TKCosmics_0T/ALCARECO/TkAlCosmics0T-0T_110X_mcRun3_2021cosmics_realistic_deco_v4-v1/40000/D50D6D4D-1688-7346-94DE-CE5D981BD4BA.root',
                            '/store/mc/Run3Winter20CosmicDR/TKCosmics_0T/ALCARECO/TkAlCosmics0T-0T_110X_mcRun3_2021cosmics_realistic_deco_v4-v1/40000/B62C9835-7E73-C84D-A6EF-8E249C1E19DB.root',
                            '/store/mc/Run3Winter20CosmicDR/TKCosmics_0T/ALCARECO/TkAlCosmics0T-0T_110X_mcRun3_2021cosmics_realistic_deco_v4-v1/40000/B632FC1E-9AAB-1642-A0D0-497B940EECF2.root',
                            '/store/mc/Run3Winter20CosmicDR/TKCosmics_0T/ALCARECO/TkAlCosmics0T-0T_110X_mcRun3_2021cosmics_realistic_deco_v4-v1/40000/7F211653-1856-D640-BC91-09E219AD4799.root',
                            '/store/mc/Run3Winter20CosmicDR/TKCosmics_0T/ALCARECO/TkAlCosmics0T-0T_110X_mcRun3_2021cosmics_realistic_deco_v4-v1/40000/740F9E5E-B251-4B46-892E-7A502E5823B1.root',
                            '/store/mc/Run3Winter20CosmicDR/TKCosmics_0T/ALCARECO/TkAlCosmics0T-0T_110X_mcRun3_2021cosmics_realistic_deco_v4-v1/40000/EF14AD41-858E-8D42-A509-7882C885D031.root',
                            '/store/mc/Run3Winter20CosmicDR/TKCosmics_0T/ALCARECO/TkAlCosmics0T-0T_110X_mcRun3_2021cosmics_realistic_deco_v4-v1/40000/B615914B-440A-934F-80A2-AF86DBC43BB2.root',
                            '/store/mc/Run3Winter20CosmicDR/TKCosmics_0T/ALCARECO/TkAlCosmics0T-0T_110X_mcRun3_2021cosmics_realistic_deco_v4-v1/40000/F3587F6B-0269-514D-876B-F0AC7F3B92F2.root',
                            '/store/mc/Run3Winter20CosmicDR/TKCosmics_0T/ALCARECO/TkAlCosmics0T-0T_110X_mcRun3_2021cosmics_realistic_deco_v4-v1/40000/E146835C-E25A-4944-A361-0A50B0495489.root',
                            '/store/mc/Run3Winter20CosmicDR/TKCosmics_0T/ALCARECO/TkAlCosmics0T-0T_110X_mcRun3_2021cosmics_realistic_deco_v4-v1/40000/F1EB8004-5B67-7F4F-BE52-2575020BF69A.root',
                            '/store/mc/Run3Winter20CosmicDR/TKCosmics_0T/ALCARECO/TkAlCosmics0T-0T_110X_mcRun3_2021cosmics_realistic_deco_v4-v1/40000/82C96A4E-38BE-0943-8C66-5DDEE37C5813.root',
                            '/store/mc/Run3Winter20CosmicDR/TKCosmics_0T/ALCARECO/TkAlCosmics0T-0T_110X_mcRun3_2021cosmics_realistic_deco_v4-v1/40000/95F6A7D4-6D6F-5942-AE12-AE736195FF6D.root',
                            '/store/mc/Run3Winter20CosmicDR/TKCosmics_0T/ALCARECO/TkAlCosmics0T-0T_110X_mcRun3_2021cosmics_realistic_deco_v4-v1/40000/D8608BF3-AD96-0241-8161-FD16BEAA5A33.root',
                            '/store/mc/Run3Winter20CosmicDR/TKCosmics_0T/ALCARECO/TkAlCosmics0T-0T_110X_mcRun3_2021cosmics_realistic_deco_v4-v1/40000/556D04C1-86E6-034F-A1A3-DD2810252BE4.root',
                            '/store/mc/Run3Winter20CosmicDR/TKCosmics_0T/ALCARECO/TkAlCosmics0T-0T_110X_mcRun3_2021cosmics_realistic_deco_v4-v1/40000/5F7D0DAF-5A80-4649-90A5-9BB1C34737A4.root',
                            '/store/mc/Run3Winter20CosmicDR/TKCosmics_0T/ALCARECO/TkAlCosmics0T-0T_110X_mcRun3_2021cosmics_realistic_deco_v4-v1/40000/5D059367-6054-3243-BAD6-35774D631788.root',
                            '/store/mc/Run3Winter20CosmicDR/TKCosmics_0T/ALCARECO/TkAlCosmics0T-0T_110X_mcRun3_2021cosmics_realistic_deco_v4-v1/40000/BCC8318D-4377-2C46-82D3-A267185E2B35.root',
                            '/store/mc/Run3Winter20CosmicDR/TKCosmics_0T/ALCARECO/TkAlCosmics0T-0T_110X_mcRun3_2021cosmics_realistic_deco_v4-v1/40000/1F000441-CAF0-8D48-93F1-CB9244D36580.root',
                            '/store/mc/Run3Winter20CosmicDR/TKCosmics_0T/ALCARECO/TkAlCosmics0T-0T_110X_mcRun3_2021cosmics_realistic_deco_v4-v1/40000/CC752AC2-C16C-534E-A894-6C9FC3F65F0C.root',
                            '/store/mc/Run3Winter20CosmicDR/TKCosmics_0T/ALCARECO/TkAlCosmics0T-0T_110X_mcRun3_2021cosmics_realistic_deco_v4-v1/40000/59ADBCB6-E46C-7940-BF09-2C8A5F153C4A.root',
                            '/store/mc/Run3Winter20CosmicDR/TKCosmics_0T/ALCARECO/TkAlCosmics0T-0T_110X_mcRun3_2021cosmics_realistic_deco_v4-v1/40000/5B775354-97FC-714B-878C-9E3C87480700.root',
                            '/store/mc/Run3Winter20CosmicDR/TKCosmics_0T/ALCARECO/TkAlCosmics0T-0T_110X_mcRun3_2021cosmics_realistic_deco_v4-v1/40000/12C6857F-D5EC-194A-8A54-58EF1C8891F2.root',
                            '/store/mc/Run3Winter20CosmicDR/TKCosmics_0T/ALCARECO/TkAlCosmics0T-0T_110X_mcRun3_2021cosmics_realistic_deco_v4-v1/40000/5118C74A-DD2A-C746-A509-5630627C8504.root',
                            '/store/mc/Run3Winter20CosmicDR/TKCosmics_0T/ALCARECO/TkAlCosmics0T-0T_110X_mcRun3_2021cosmics_realistic_deco_v4-v1/40000/2B112674-12D1-CB42-A59B-1B4CD5A217CA.root',
                            '/store/mc/Run3Winter20CosmicDR/TKCosmics_0T/ALCARECO/TkAlCosmics0T-0T_110X_mcRun3_2021cosmics_realistic_deco_v4-v1/40000/F9E86B93-D295-574A-8724-AE458336052E.root',
                            '/store/mc/Run3Winter20CosmicDR/TKCosmics_0T/ALCARECO/TkAlCosmics0T-0T_110X_mcRun3_2021cosmics_realistic_deco_v4-v1/40000/893EAAE8-EA4B-D64E-A82E-11D96D2CBFC9.root',
                            '/store/mc/Run3Winter20CosmicDR/TKCosmics_0T/ALCARECO/TkAlCosmics0T-0T_110X_mcRun3_2021cosmics_realistic_deco_v4-v1/40000/394A3448-D279-DC4F-82BD-E994C17E9320.root',
                            '/store/mc/Run3Winter20CosmicDR/TKCosmics_0T/ALCARECO/TkAlCosmics0T-0T_110X_mcRun3_2021cosmics_realistic_deco_v4-v1/40000/980676CB-1F78-F54F-8E76-A33FC1B6F368.root',
                            '/store/mc/Run3Winter20CosmicDR/TKCosmics_0T/ALCARECO/TkAlCosmics0T-0T_110X_mcRun3_2021cosmics_realistic_deco_v4-v1/40000/093743EF-C992-BE45-8EC6-C3A58EE67058.root',
                            '/store/mc/Run3Winter20CosmicDR/TKCosmics_0T/ALCARECO/TkAlCosmics0T-0T_110X_mcRun3_2021cosmics_realistic_deco_v4-v1/40000/10142C68-2781-CE4A-A04A-49016F95753F.root',
                            '/store/mc/Run3Winter20CosmicDR/TKCosmics_0T/ALCARECO/TkAlCosmics0T-0T_110X_mcRun3_2021cosmics_realistic_deco_v4-v1/40000/A9A98929-657C-3747-828A-93416C3C990E.root',
                            '/store/mc/Run3Winter20CosmicDR/TKCosmics_0T/ALCARECO/TkAlCosmics0T-0T_110X_mcRun3_2021cosmics_realistic_deco_v4-v1/40000/D19D8078-9EA1-4649-BACC-2F0559B1E88E.root',
                            '/store/mc/Run3Winter20CosmicDR/TKCosmics_0T/ALCARECO/TkAlCosmics0T-0T_110X_mcRun3_2021cosmics_realistic_deco_v4-v1/40000/FCFB56CF-8C56-C64C-B672-139648F32E91.root',
                            '/store/mc/Run3Winter20CosmicDR/TKCosmics_0T/ALCARECO/TkAlCosmics0T-0T_110X_mcRun3_2021cosmics_realistic_deco_v4-v1/40000/44D2D421-9B16-FA47-9C53-B8B93A7F2077.root',
                            '/store/mc/Run3Winter20CosmicDR/TKCosmics_0T/ALCARECO/TkAlCosmics0T-0T_110X_mcRun3_2021cosmics_realistic_deco_v4-v1/40000/525F1D6D-9271-FF41-B05E-C99FFCB029D9.root',
                            '/store/mc/Run3Winter20CosmicDR/TKCosmics_0T/ALCARECO/TkAlCosmics0T-0T_110X_mcRun3_2021cosmics_realistic_deco_v4-v1/40000/39381C24-97B6-E641-B4AB-7D1B1D25A782.root',
                            '/store/mc/Run3Winter20CosmicDR/TKCosmics_0T/ALCARECO/TkAlCosmics0T-0T_110X_mcRun3_2021cosmics_realistic_deco_v4-v1/40000/08901961-CFB6-4A43-98FC-D1817EF76D13.root'
]
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(options.maxEvents))


#readFiles = cms.untracked.vstring()
#process.source = cms.Source("PoolSource",
#                            fileNames = readFiles)
# the_files=[]
# file_list = glob.glob("/eos/cms/store/group/alca_trackeralign/pkeicher/test_out/CosmicsRun3MCProduction/2021/*")
# for f in file_list:
#     the_files.append(f.replace("/eos/cms",""))
# readFiles.extend(the_files)
    
# print(the_files)
# process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(10000))

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

process.myanalysis2 = cms.EDAnalyzer("DMRChecker_v2",
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
                      process.myanalysis2
                      )
