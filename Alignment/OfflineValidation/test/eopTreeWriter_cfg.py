import FWCore.ParameterSet.Config as cms

process = cms.Process("EnergyOverMomentumTree")

# initialize MessageLogger and output report
process.load("FWCore.MessageService.MessageLogger_cfi")
#process.MessageLogger.cerr.threshold = 'ERROR'
process.MessageLogger.cerr.FwkReport.reportEvery = 1000
process.MessageLogger.TrackRefitter=dict()

process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(False) )

# define input files
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
'/store/relval/CMSSW_11_3_0_pre3/RelValMinBias_13/GEN-SIM-RECO/113X_upgrade2018_realistic_v3-v1/00000/00971a25-0d77-4a54-8b38-4ee45f26ac27.root',
'/store/relval/CMSSW_11_3_0_pre3/RelValMinBias_13/GEN-SIM-RECO/113X_upgrade2018_realistic_v3-v1/00000/014ac00e-38df-45ac-9501-82944400515e.root',
'/store/relval/CMSSW_11_3_0_pre3/RelValMinBias_13/GEN-SIM-RECO/113X_upgrade2018_realistic_v3-v1/00000/05a0879c-eca2-4a7e-89c3-f0f20a05c73d.root',
'/store/relval/CMSSW_11_3_0_pre3/RelValMinBias_13/GEN-SIM-RECO/113X_upgrade2018_realistic_v3-v1/00000/05c96563-4f60-40ad-9eb3-cfdd2dad1c09.root',
'/store/relval/CMSSW_11_3_0_pre3/RelValMinBias_13/GEN-SIM-RECO/113X_upgrade2018_realistic_v3-v1/00000/72009661-4aa4-4a0d-84f7-f1279dadffe4.root'
    )
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

# load configuration files
####################################################################
# Get the Magnetic Field
####################################################################
process.load('Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff')
process.load("Configuration.StandardSequences.MagneticField_cff")

from Configuration.Geometry.GeometryRecoDB_cff import *
process.load("Configuration.Geometry.GeometryRecoDB_cff")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")

#process.GlobalTag.globaltag = '113X_dataRun2_v6'
import FWCore.ParameterSet.VarParsing as VarParsing
options = VarParsing.VarParsing("analysis")

options.register ('GlobalTag',
                  'auto:phase1_2021_realistic',
                  VarParsing.VarParsing.multiplicity.singleton,  # singleton or list
                  VarParsing.VarParsing.varType.string,          # string, int, or float
                  "Global Tag to be used")

options.parseArguments()

print( "conditionGT       : ", options.GlobalTag)
print( "maxEvents         : ", options.maxEvents)


from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, options.GlobalTag, '')

## jsonFile =  'json_Apr21_May10v2_Promptv4_136035_167913.txt'
## import PhysicsTools.PythonAnalysis.LumiList as LumiList
## import FWCore.ParameterSet.Types as CfgTypes
## print "JSON used: ", jsonFile
## myLumis = LumiList.LumiList(filename = jsonFile).getCMSSWString().split(',')
## process.source.lumisToProcess = CfgTypes.untracked(CfgTypes.VLuminosityBlockRange())
## process.source.lumisToProcess.extend(myLumis)

process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")

#process.load("Geometry.CaloEventSetup.CaloGeometry_cff")

process.load("Geometry.CaloEventSetup.CaloTopology_cfi")

process.load("Geometry.CaloEventSetup.EcalTrigTowerConstituents_cfi")

#process.load("Geometry.TrackerGeometryBuilder.trackerGeometry_cfi")

#process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")

process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")

process.load("Geometry.DTGeometry.dtGeometry_cfi")

process.load("Geometry.RPCGeometry.rpcGeometry_cfi")

process.load("Geometry.CSCGeometry.cscGeometry_cfi")

process.load("Geometry.CommonTopologies.bareGlobalTrackingGeometry_cfi")
#from TrackingTools.TrackAssociator.default_cfi import *
process.load("TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagator_cfi")
process.load("TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAny_cfi")
process.load("TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAlong_cfi")
process.load("TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorOpposite_cfi")
process.load("TrackingTools.TrackAssociator.DetIdAssociatorESProducer_cff")
from TrackingTools.TrackAssociator.default_cfi import *
#process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_noesprefer_cff")

# choose geometry
from CondCore.DBCommon.CondDBSetup_cfi import *
process.trackerAlignment = cms.ESSource("PoolDBESSource",
					CondDBSetup,
                                        #connect = cms.string("frontier://FrontierArc/CMS_COND_31X_ALIGNMENT_BD19"),
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
                                              			#tag = cms.string("TrackerAlignmentExtendedErrors_v16_offline_IOVs")
                                                             )
                                                    ),
                                      connect = cms.string("frontier://FrontierProd/CMS_CONDITIONS")
    )
process.es_prefer_TrackerAPE = cms.ESPrefer("PoolDBESSource", "trackerAPE")
#process.prefer("trackerAlignment")

process.load("RecoVertex.BeamSpotProducer.BeamSpot_cfi")

# configure alignment track selector
process.load("Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi")
process.AlignmentTrackSelector.src = cms.InputTag('generalTracks:') # 'TkAlIsoProd' # trackCollection' # 'ALCARECOTkAlZMuMu' # 'ALCARECOTkAlMinBias' # adjust to input file
process.AlignmentTrackSelector.ptMin = 1.
process.AlignmentTrackSelector.etaMin = -5.
process.AlignmentTrackSelector.etaMax = 5.
process.AlignmentTrackSelector.nHitMin = 5
process.AlignmentTrackSelector.chi2nMax = 100.
#process.AlignmentTrackSelector.applyNHighestPt = True
#process.AlignmentTrackSelector.nHighestPt = 2

# configure track refitter
import RecoTracker.TrackProducer.TrackRefitters_cff
process.load("RecoTracker.TrackProducer.TrackRefitters_cff")
process.load("RecoTracker.MeasurementDet.MeasurementTrackerEventProducer_cfi")
process.TrackRefitter.src = cms.InputTag('generalTracks')
#process.TrackRefitter.src = cms.InputTag('AlignmentTrackSelector')
process.TrackRefitter.TrajectoryInEvent = True

# configure tree writer
#TrackAssociatorParameterBlock.TrackAssociatorParameters.EERecHitCollectionLabel = cms.InputTag("IsoProd","IsoTrackEcalRecHitCollection")
TrackAssociatorParameterBlock.TrackAssociatorParameters.EERecHitCollectionLabel = cms.InputTag("ecalRecHit","EcalRecHitsEE")
#TrackAssociatorParameterBlock.TrackAssociatorParameters.EBRecHitCollectionLabel = cms.InputTag("IsoProd","IsoTrackEcalRecHitCollection")
TrackAssociatorParameterBlock.TrackAssociatorParameters.EBRecHitCollectionLabel = cms.InputTag("ecalRecHit","EcalRecHitsEB")
#TrackAssociatorParameterBlock.TrackAssociatorParameters.HBHERecHitCollectionLabel = cms.InputTag("IsoProd","IsoTrackHBHERecHitCollection")
#TrackAssociatorParameterBlock.TrackAssociatorParameters.HORecHitCollectionLabel = cms.InputTag("IsoProd","IsoTrackHORecHitCollection")

process.energyOverMomentumTree = cms.EDAnalyzer('EopTreeWriter',
    TrackAssociatorParameterBlock
)
process.energyOverMomentumTree.src = cms.InputTag('TrackRefitter')
#process.energyOverMomentumTree.src = cms.InputTag('TkAlIsoProd:')

process.TFileService = cms.Service("TFileService",
    fileName = cms.string('test_EopTree.root')
)

process.p = cms.Path(process.MeasurementTrackerEvent*process.offlineBeamSpot*process.TrackRefitter*process.energyOverMomentumTree)
#process.p = cms.Path(process.offlineBeamSpot*process.AlignmentTrackSelector*process.TrackRefitter*process.energyOverMomentumTree)
