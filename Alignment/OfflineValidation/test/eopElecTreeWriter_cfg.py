REFIT = True
MISALIGN = False

if not REFIT and not MISALIGN:
    print( "Normal mode")
elif REFIT and not MISALIGN:
    print( "REFIT only MODE")
elif REFIT and MISALIGN:
    print( "REFIT + MISALIGN")
else :
    print( "ERROR! STOP!")
    exit
    
import FWCore.ParameterSet.VarParsing as VarParsing
options = VarParsing.VarParsing("analysis")

options.register ('outputRootFile',
                  "test_EopTreeElectron.root",
                  VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                  VarParsing.VarParsing.varType.string,         # string, int, or float
                  "output root file")

options.register ('GlobalTag',
                  'auto:phase1_2022_realistic',
                  VarParsing.VarParsing.multiplicity.singleton,  # singleton or list
                  VarParsing.VarParsing.varType.string,          # string, int, or float
                  "Global Tag to be used")

options.parseArguments()

print( "conditionGT       : ", options.GlobalTag)
print( "outputFile        : ", options.outputRootFile)
print( "maxEvents         : ", options.maxEvents)

import FWCore.ParameterSet.Config as cms
from Configuration.Eras.Era_Run3_cff import Run3
process = cms.Process("EnergyOverMomentumTree",Run3)
    
#process.Tracer = cms.Service("Tracer")

# initialize MessageLogger and output report
process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 100
process.MessageLogger.TrackRefitter=dict()
process.MessageLogger.EopElecTreeWriter=dict()
    
process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(False))
    
# define input files
process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(
                                '/store/relval/CMSSW_12_4_0_pre4/RelValZEE_14/GEN-SIM-RECO/PU_124X_mcRun3_2021_realistic_v1-v1/2580000/4a1ae43b-f4b3-4ad9-b86e-a7d9f6fc5c40.root',
                                '/store/relval/CMSSW_12_4_0_pre4/RelValZEE_14/GEN-SIM-RECO/PU_124X_mcRun3_2021_realistic_v1-v1/2580000/33565608-3cac-47fe-a1fc-aef60f866b3a.root',
                                '/store/relval/CMSSW_12_4_0_pre4/RelValZEE_14/GEN-SIM-RECO/PU_124X_mcRun3_2021_realistic_v1-v1/2580000/87fa96e1-925f-4cd3-878d-98a735737e55.root',
                                '/store/relval/CMSSW_12_4_0_pre4/RelValZEE_14/GEN-SIM-RECO/PU_124X_mcRun3_2021_realistic_v1-v1/2580000/ea3a1cc8-720f-4392-9f0b-bd04d7f236a8.root',
                                '/store/relval/CMSSW_12_4_0_pre4/RelValZEE_14/GEN-SIM-RECO/PU_124X_mcRun3_2021_realistic_v1-v1/2580000/3b3e7330-6174-43f3-8a49-c12eeae4d7f2.root',
                                '/store/relval/CMSSW_12_4_0_pre4/RelValZEE_14/GEN-SIM-RECO/PU_124X_mcRun3_2021_realistic_v1-v1/2580000/ddc48c68-781c-485b-887e-4fcd6f0e0772.root',
                                '/store/relval/CMSSW_12_4_0_pre4/RelValZEE_14/GEN-SIM-RECO/PU_124X_mcRun3_2021_realistic_v1-v1/2580000/a366a7ca-b71c-457a-8b64-09040f5b5819.root',
                                '/store/relval/CMSSW_12_4_0_pre4/RelValZEE_14/GEN-SIM-RECO/PU_124X_mcRun3_2021_realistic_v1-v1/2580000/4758604c-b0c1-4e09-a0d4-38dd0da16789.root',
                                '/store/relval/CMSSW_12_4_0_pre4/RelValZEE_14/GEN-SIM-RECO/PU_124X_mcRun3_2021_realistic_v1-v1/2580000/48985150-6f47-4a7f-b09f-b6301b7ec6ff.root'))
                        
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(options.maxEvents)
)

####################################################################
# Load  electron configuration files
####################################################################
process.load("TrackingTools.GsfTracking.GsfElectronFit_cff")

####################################################################
# Get the Magnetic Field
####################################################################
process.load('Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff')

###################################################################
# Standard loads
###################################################################
from Configuration.Geometry.GeometryRecoDB_cff import *
process.load("Configuration.Geometry.GeometryRecoDB_cff")

####################################################################
# Get the BeamSpot
####################################################################
process.load("RecoVertex.BeamSpotProducer.BeamSpot_cff")

####################################################################
# Get the GlogalTag
####################################################################
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, options.GlobalTag, '')

# choose geometry
if MISALIGN:
    print( "MISALIGN")
    from CondCore.CondDB.CondDB_cfi import CondDB
    CondDB.connect = cms.string("frontier://FrontierProd/CMS_CONDITIONS")
    process.trackerAlignment = cms.ESSource("PoolDBESSource",
                                            CondDB,
                                            toGet = cms.VPSet(cms.PSet(record = cms.string("TrackerAlignmentRcd"),
                                                                       tag = cms.string("TrackerAlignment_2017_ultralegacymc_v2")
                                                                       )
                                                          )
    )
    process.es_prefer_trackerAlignment = cms.ESPrefer("PoolDBESSource", "trackerAlignment")

    process.trackerAPE = cms.ESSource("PoolDBESSource",
                                      CondDB,
                                      toGet = cms.VPSet(cms.PSet(record = cms.string("TrackerAlignmentErrorRcd"),
                                                                 tag = cms.string("TrackerAlignmentExtendedErrors_2017_ultralegacymc_v2")
                                                                 )
                                                        )
    )
    process.es_prefer_TrackerAPE = cms.ESPrefer("PoolDBESSource", "trackerAPE")
    
else:
    print( "NO MISALIGN")
    
# configure Gsf track refitter
if REFIT:
    print( "REFIT")
    process.load("RecoTracker.TrackProducer.TrackRefitters_cff")
    import RecoTracker.TrackProducer.TrackRefitters_cff
    process.load("TrackingTools.TransientTrack.TransientTrackBuilder_cfi")
    process.load("TrackingTools.GsfTracking.fwdGsfElectronPropagator_cff")
    process.load("RecoTracker.TrackProducer.GsfTrackRefitter_cff")
    process.GsfTrackRefitter.src = cms.InputTag('electronGsfTracks')  
    process.GsfTrackRefitter.TrajectoryInEvent = True
    process.GsfTrackRefitter.AlgorithmName = cms.string('gsf')
else:
    print( "NO REFIT")

process.load("Alignment.OfflineValidation.eopElecTreeWriter_cfi")

if REFIT:
    print( "REFIT")
    process.energyOverMomentumTree.src = cms.InputTag('GsfTrackRefitter')
else:
    print( "NO REFIT")
    process.energyOverMomentumTree.src = cms.InputTag('electronGsfTracks')
     
process.TFileService = cms.Service("TFileService",
                                   fileName = cms.string(options.outputRootFile)
                                   )
 
if REFIT:
    print( "REFIT")
    process.p = cms.Path(process.offlineBeamSpot
                         *process.GsfTrackRefitter
                         *process.energyOverMomentumTree)
else:
    print( "NO REFIT")
    process.p = cms.Path(process.offlineBeamSpot
                         *process.energyOverMomentumTree)
