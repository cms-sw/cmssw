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
import Configuration.Geometry.defaultPhase2ConditionsEra_cff as _settings
_PH2_GLOBAL_TAG, _PH2_ERA = _settings.get_era_and_conditions(_settings.DEFAULT_VERSION)


options = VarParsing.VarParsing("analysis")

options.register ('outputRootFile',
                  "test_EopTreeElectron.root",
                  VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                  VarParsing.VarParsing.varType.string,         # string, int, or float
                  "output root file")

options.register ('GlobalTag',
                  _PH2_GLOBAL_TAG,
                  VarParsing.VarParsing.multiplicity.singleton,  # singleton or list
                  VarParsing.VarParsing.varType.string,          # string, int, or float
                  "Global Tag to be used")

options.parseArguments()

print( "conditionGT       : ", options.GlobalTag)
print( "outputFile        : ", options.outputRootFile)
print( "maxEvents         : ", options.maxEvents)


import FWCore.ParameterSet.Config as cms
process = cms.Process("EnergyOverMomentumTree", _PH2_ERA)
    
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
                                '/store/relval/CMSSW_20_0_0_pre1/RelValZEE_14/GEN-SIM-RECO/150X_mcRun4_realistic_v1_STD_RegeneratedGS_D121_noPU-v1/2590000/028e09cc-784b-41ab-a712-511d5bb67724.root',
                                '/store/relval/CMSSW_20_0_0_pre1/RelValZEE_14/GEN-SIM-RECO/150X_mcRun4_realistic_v1_STD_RegeneratedGS_D121_noPU-v1/2590000/b1273478-d951-4915-b888-c5e73d49f39c.root',
                                '/store/relval/CMSSW_20_0_0_pre1/RelValZEE_14/GEN-SIM-RECO/150X_mcRun4_realistic_v1_STD_RegeneratedGS_D121_noPU-v1/2590000/649c30e1-8ec6-4e31-9f17-4c1dd1f6409d.root',
                                '/store/relval/CMSSW_20_0_0_pre1/RelValZEE_14/GEN-SIM-RECO/150X_mcRun4_realistic_v1_STD_RegeneratedGS_D121_noPU-v1/2590000/fd8de856-99a9-477c-bb0d-cd572582b004.root',
                                '/store/relval/CMSSW_20_0_0_pre1/RelValZEE_14/GEN-SIM-RECO/150X_mcRun4_realistic_v1_STD_RegeneratedGS_D121_noPU-v1/2590000/5aa81899-8d7d-46c9-8636-667de1facc9a.root',
                                '/store/relval/CMSSW_20_0_0_pre1/RelValZEE_14/GEN-SIM-RECO/150X_mcRun4_realistic_v1_STD_RegeneratedGS_D121_noPU-v1/2590000/af539472-e507-4896-bc6b-c9795e2def16.root',
                                '/store/relval/CMSSW_20_0_0_pre1/RelValZEE_14/GEN-SIM-RECO/150X_mcRun4_realistic_v1_STD_RegeneratedGS_D121_noPU-v1/2590000/0abaf7aa-c2ec-4bcb-96f1-d22d5c7bd937.root',
                                '/store/relval/CMSSW_20_0_0_pre1/RelValZEE_14/GEN-SIM-RECO/150X_mcRun4_realistic_v1_STD_RegeneratedGS_D121_noPU-v1/2590000/053a2340-040a-4be4-b335-7f546f24bab6.root',
                                '/store/relval/CMSSW_20_0_0_pre1/RelValZEE_14/GEN-SIM-RECO/150X_mcRun4_realistic_v1_STD_RegeneratedGS_D121_noPU-v1/2590000/cf187c8c-96e7-4949-9a75-67d4b7696cf8.root',
                                '/store/relval/CMSSW_20_0_0_pre1/RelValZEE_14/GEN-SIM-RECO/150X_mcRun4_realistic_v1_STD_RegeneratedGS_D121_noPU-v1/2590000/0c2bfdc1-278c-4e60-8fa1-2ef58cc3a35b.root'),
                            # Workaround for backwards-incompatible change in these types.
                            # Can be removed after the input file(s) have been updated to something more recent than 20_1_0_pre2 RelVals
                            inputCommands = cms.untracked.vstring([
                                "keep *",
                                "drop Phase2TrackerDigiedmDetSetVectorPhase2TrackerDigiPhase2TrackerDigiedmrefhelperFindForDetSetVectoredmRefTTClusterAssociationMap_*_*_*",
                                "drop Phase2TrackerDigiedmDetSetVectorPhase2TrackerDigiPhase2TrackerDigiedmrefhelperFindForDetSetVectoredmRefTTClusteredmNewDetSetVector_*_*_*",
                                "drop Phase2TrackerDigiedmDetSetVectorPhase2TrackerDigiPhase2TrackerDigiedmrefhelperFindForDetSetVectoredmRefTTStubAssociationMap_*_*_*",
                                "drop Phase2TrackerDigiedmDetSetVectorPhase2TrackerDigiPhase2TrackerDigiedmrefhelperFindForDetSetVectoredmRefTTStubedmNewDetSetVector_*_*_*",
                                "drop Phase2TrackerDigiedmDetSetVectorPhase2TrackerDigiPhase2TrackerDigiedmrefhelperFindForDetSetVectoredmRefTTTrackAssociationMap_*_*_*",
                            ]),
)
                        
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
process.load('Configuration.Geometry.GeometryExtendedRun4DefaultReco_cff')

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
    # TO-DO develop the Phase-2 version of these scenarios!
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

process.load("Alignment.OfflineValidation.energyOverMomentumTreeElec_cfi")

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
