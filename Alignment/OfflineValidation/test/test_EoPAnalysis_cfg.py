REFIT = True
MISALIGN = False

if not REFIT and not MISALIGN:
    print "Normal mode"
elif REFIT and not MISALIGN:
    print "REFIT only MODE"
elif REFIT and MISALIGN:
    print "REFIT + MISALIGN"
else :
    print "ERROR! STOP!"
    exit
    
import FWCore.ParameterSet.VarParsing as VarParsing
options = VarParsing.VarParsing("analysis")

options.register ('outputRootFile',
                  "test_EOverP.root",
                  VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                  VarParsing.VarParsing.varType.string,         # string, int, or float
                  "output root file")

options.register ('GlobalTag',
                  'auto:phase1_2021_realistic',
                  VarParsing.VarParsing.multiplicity.singleton,  # singleton or list
                  VarParsing.VarParsing.varType.string,          # string, int, or float
                  "Global Tag to be used")

options.parseArguments()

print "conditionGT       : ", options.GlobalTag
print "outputFile        : ", options.outputRootFile
print "maxEvents         : ", options.maxEvents

import FWCore.ParameterSet.Config as cms
process = cms.Process("EnergyOverMomentumTree")
    
# initialize MessageLogger and output report
process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 1000
process.MessageLogger.categories.append('TrackRefitter')
    
process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(False))
    
# define input files
process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(
                                '/store/relval/CMSSW_11_0_0_patch1/RelValZEE_14/GEN-SIM-RECO/PU_110X_mcRun3_2021_realistic_v6-v1/20000/40A29188-FF71-7C47-95FE-08785E537771.root',
                                '/store/relval/CMSSW_11_0_0_patch1/RelValZEE_14/GEN-SIM-RECO/PU_110X_mcRun3_2021_realistic_v6-v1/20000/7CF278C4-12CD-F040-BF72-1762962CE462.root',
                                '/store/relval/CMSSW_11_0_0_patch1/RelValZEE_14/GEN-SIM-RECO/PU_110X_mcRun3_2021_realistic_v6-v1/20000/13962B5C-0A48-124F-BC96-F36AD105FA6D.root',
                                '/store/relval/CMSSW_11_0_0_patch1/RelValZEE_14/GEN-SIM-RECO/PU_110X_mcRun3_2021_realistic_v6-v1/20000/DEA8B720-65E0-DC41-ACE0-CFF13279C27C.root',
                                '/store/relval/CMSSW_11_0_0_patch1/RelValZEE_14/GEN-SIM-RECO/PU_110X_mcRun3_2021_realistic_v6-v1/20000/88395EE9-0488-1C42-B90E-4907CDB0C006.root',
                                '/store/relval/CMSSW_11_0_0_patch1/RelValZEE_14/GEN-SIM-RECO/PU_110X_mcRun3_2021_realistic_v6-v1/20000/632A8DE3-F013-2345-9AF4-BBFC86063FD6.root',
                                '/store/relval/CMSSW_11_0_0_patch1/RelValZEE_14/GEN-SIM-RECO/PU_110X_mcRun3_2021_realistic_v6-v1/20000/65F7BE52-5536-A64E-8CB7-C177A96E5133.root',
                                '/store/relval/CMSSW_11_0_0_patch1/RelValZEE_14/GEN-SIM-RECO/PU_110X_mcRun3_2021_realistic_v6-v1/20000/B5AC0FDE-0CB1-C443-9390-97EA55494923.root',
                                '/store/relval/CMSSW_11_0_0_patch1/RelValZEE_14/GEN-SIM-RECO/PU_110X_mcRun3_2021_realistic_v6-v1/20000/78B0EC4E-2ECE-7842-872C-20466D17B36E.root'
                                #'/store/relval/CMSSW_11_0_0_pre7/RelValZEE_13/GEN-SIM-RECO/PUpmx25ns_110X_mc2017_realistic_v1_rsb-v1/10000/CD0B4494-A24B-CB46-B4B7-80A523EA5D59.root')
                            ))
                        
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
    print "MISALIGN"
    from CondCore.DBCommon.CondDBSetup_cfi import *
    process.trackerAlignment = cms.ESSource("PoolDBESSource",
                                            CondDBSetup,
                                            toGet = cms.VPSet(cms.PSet(record = cms.string("TrackerAlignmentRcd"),
                                                                       tag = cms.string("TrackerAlignment_2010Realistic_mc")
                                                                   )
                                                               ),
                                            connect = cms.string("sqlite_file:twist1p5.db")
                                            )
    process.es_prefer_trackerAlignment = cms.ESPrefer("PoolDBESSource", "trackerAlignment")

    process.trackerAPE = cms.ESSource("PoolDBESSource",CondDBSetup,
                                      toGet = cms.VPSet(cms.PSet(record = cms.string("TrackerAlignmentErrorRcd"),
                                                                 tag = cms.string("TrackerAlignmentErrors_2010Realistic_mc")
                                                             )
                                                    ),
                                      connect = cms.string("sqlite_file:twist1p5.db")
    )
    process.es_prefer_TrackerAPE = cms.ESPrefer("PoolDBESSource", "trackerAPE")
    
else:
    print "NO MISALIGN"
    
# configure Gsf track refitter
if REFIT:
    print "REFIT"
    process.load("RecoTracker.TrackProducer.TrackRefitters_cff")
    import RecoTracker.TrackProducer.TrackRefitters_cff
    process.load("TrackingTools.TransientTrack.TransientTrackBuilder_cfi")
    process.load("TrackingTools.GsfTracking.fwdGsfElectronPropagator_cff")
    process.load("RecoTracker.TrackProducer.GsfTrackRefitter_cff")
    process.GsfTrackRefitter.src = cms.InputTag('electronGsfTracks')  
    process.GsfTrackRefitter.TrajectoryInEvent = True
    process.GsfTrackRefitter.AlgorithmName = cms.string('gsf')
else:
    print "NO REFIT"

process.load("Alignment.OfflineValidation.eopElecTreeWriter_cfi")

if REFIT:
    print "REFIT"
    process.energyOverMomentumTree.src = cms.InputTag('GsfTrackRefitter')
else:
    print "NO REFIT"
    process.energyOverMomentumTree.src = cms.InputTag('electronGsfTracks')
     
process.TFileService = cms.Service("TFileService",
                                   fileName = cms.string(options.outputRootFile)
                                   )
 
if REFIT:
    print "REFIT"
    process.p = cms.Path(process.offlineBeamSpot*
                         process.GsfTrackRefitter*
                         process.energyOverMomentumTree)    
else:
    print "NO REFIT"
    process.p = cms.Path(process.offlineBeamSpot*
                         process.energyOverMomentumTree)
