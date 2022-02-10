REFIT = True
MISALIGN = True

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
                  "test_EOverP_Electrons.root",
                  VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                  VarParsing.VarParsing.varType.string,         # string, int, or float
                  "output root file")

options.register ('GlobalTag',
                  'auto:phase1_2021_realistic',
                  VarParsing.VarParsing.multiplicity.singleton,  # singleton or list
                  VarParsing.VarParsing.varType.string,          # string, int, or float
                  "Global Tag to be used")

options.parseArguments()

print( "conditionGT       : ", options.GlobalTag)
print( "outputFile        : ", options.outputRootFile)
print( "maxEvents         : ", options.maxEvents)

import FWCore.ParameterSet.Config as cms
process = cms.Process("EnergyOverMomentumTree")
    
# initialize MessageLogger and output report
process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 1000
process.MessageLogger.TrackRefitter=dict()
    
process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(False))
    
# define input files
process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(
                                '/store/relval/CMSSW_10_5_0/RelValZEE_13/GEN-SIM-RECO/PU25ns_105X_upgrade2018_realistic_EcalHcal_v2_HS-v1/10000/03CA556E-7056-394F-A105-5270966A8CF6.root',
'/store/relval/CMSSW_10_5_0/RelValZEE_13/GEN-SIM-RECO/PU25ns_105X_upgrade2018_realistic_EcalHcal_v2_HS-v1/10000/8E8874B1-A3BC-9F4A-AF74-AE65F9B6634D.root',
'/store/relval/CMSSW_10_5_0/RelValZEE_13/GEN-SIM-RECO/PU25ns_105X_upgrade2018_realistic_EcalHcal_v2_HS-v1/10000/06228194-EA17-354B-9114-89E67DEF2131.root',
'/store/relval/CMSSW_10_5_0/RelValZEE_13/GEN-SIM-RECO/PU25ns_105X_upgrade2018_realistic_EcalHcal_v2_HS-v1/10000/DE56DFA4-DBF1-3D4A-A9F3-50ADDCFF3D93.root',
'/store/relval/CMSSW_10_5_0/RelValZEE_13/GEN-SIM-RECO/PU25ns_105X_upgrade2018_realistic_EcalHcal_v2_HS-v1/10000/FE109BAC-006F-4E43-890F-CFD14491E207.root'))
                        
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
    from CondCore.DBCommon.CondDBSetup_cfi import *
    process.trackerAlignment = cms.ESSource("PoolDBESSource",
                                            CondDBSetup,
                                            toGet = cms.VPSet(cms.PSet(record = cms.string("TrackerAlignmentRcd"),
									tag = cms.string("TrackerAlignment_2017_ultralegacymc_v2")
                                                                   )
                                                               ),
					connect = cms.string("frontier://FrontierProd/CMS_CONDITIONS")
                                            )
    process.es_prefer_trackerAlignment = cms.ESPrefer("PoolDBESSource", "trackerAlignment")

    process.trackerAPE = cms.ESSource("PoolDBESSource",CondDBSetup,
                                      toGet = cms.VPSet(cms.PSet(record = cms.string("TrackerAlignmentErrorRcd"),
								tag = cms.string("TrackerAlignmentExtendedErrors_2017_ultralegacymc_v2")
                                                             )
                                                    ),
				connect = cms.string("frontier://FrontierProd/CMS_CONDITIONS")
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
    process.p = cms.Path(process.offlineBeamSpot*
                         process.GsfTrackRefitter*
                         process.energyOverMomentumTree)    
else:
    print( "NO REFIT")
    process.p = cms.Path(process.offlineBeamSpot*
                         process.energyOverMomentumTree)
