import FWCore.ParameterSet.Config as cms

process = cms.Process("EnergyOverMomentumTree")

# initialize MessageLogger and output report
process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.threshold = 'ERROR'
process.MessageLogger.cerr.FwkReport.reportEvery = 10000
process.MessageLogger.categories.append('TrackRefitter')

process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(False) )

# define input files
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
    '/store/data/Run2011A/Commissioning/ALCARECO/HcalCalIsoTrk-May10ReReco-v2/0000/FEB81703-0E7F-E011-8161-0025B3E05E00.root'
    )
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

# load configuration files
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.Geometry.GeometryRecoDB_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'GR_R_42_V14::All'

## jsonFile =  'json_Apr21_May10v2_Promptv4_136035_167913.txt'
## import PhysicsTools.PythonAnalysis.LumiList as LumiList
## import FWCore.ParameterSet.Types as CfgTypes
## print "JSON used: ", jsonFile
## myLumis = LumiList.LumiList(filename = jsonFile).getCMSSWString().split(',')
## process.source.lumisToProcess = CfgTypes.untracked(CfgTypes.VLuminosityBlockRange())
## process.source.lumisToProcess.extend(myLumis)

process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")

process.load("Geometry.CaloEventSetup.CaloGeometry_cff")

process.load("Geometry.CaloEventSetup.CaloTopology_cfi")

process.load("Geometry.CaloEventSetup.EcalTrigTowerConstituents_cfi")

process.load("Geometry.TrackerGeometryBuilder.trackerGeometry_cfi")

process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")

process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")

process.load("Geometry.DTGeometry.dtGeometry_cfi")

process.load("Geometry.RPCGeometry.rpcGeometry_cfi")

process.load("Geometry.CSCGeometry.cscGeometry_cfi")

process.load("Geometry.CommonDetUnit.bareGlobalTrackingGeometry_cfi")
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
process.trackerAlignment = cms.ESSource("PoolDBESSource",CondDBSetup,
                                        #connect = cms.string("frontier://FrontierArc/CMS_COND_31X_ALIGNMENT_AK25"),
                                        connect = cms.string("frontier://FrontierArc/CMS_COND_31X_ALIGNMENT_BD19"),
                                        toGet = cms.VPSet(cms.PSet(record = cms.string("TrackerAlignmentRcd"),
                                                                   tag = cms.string("TrackerAlignment_GR10_v4_offline")
                                                                   )## ,
##                                                           cms.PSet(record = cms.string("TrackerAlignmentErrorExtendedRcd"),
##                                                                    tag = cms.string("TrackerAlignmentErrorsExtended_GR10_v2_offline")
##                                                                    ),
##                                                           cms.PSet(record = cms.string("TrackerSurfaceDeformationRcd"),
##                                                                    tag = cms.string("Deformations")
##                                                                    )
                                                          )
                                        )
process.prefer("trackerAlignment")

process.load("RecoVertex.BeamSpotProducer.BeamSpot_cfi")

# configure alignment track selector
process.load("Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi")
process.AlignmentTrackSelector.src = cms.InputTag('TkAlIsoProd:') # 'generalTracks' # trackCollection' # 'ALCARECOTkAlZMuMu' # 'ALCARECOTkAlMinBias' # adjust to input file
process.AlignmentTrackSelector.ptMin = 1.
process.AlignmentTrackSelector.etaMin = -5.
process.AlignmentTrackSelector.etaMax = 5.
process.AlignmentTrackSelector.nHitMin = 5
process.AlignmentTrackSelector.chi2nMax = 100.
#process.AlignmentTrackSelector.applyNHighestPt = True
#process.AlignmentTrackSelector.nHighestPt = 2

# configure track refitter
process.load("RecoTracker.TrackProducer.TrackRefitters_cff")
process.TrackRefitter.src = cms.InputTag('TkAlIsoProd:')
#process.TrackRefitter.src = cms.InputTag('AlignmentTrackSelector')
process.TrackRefitter.TrajectoryInEvent = True

# configure tree writer
TrackAssociatorParameterBlock.TrackAssociatorParameters.EERecHitCollectionLabel = cms.InputTag("IsoProd","IsoTrackEcalRecHitCollection")
TrackAssociatorParameterBlock.TrackAssociatorParameters.EBRecHitCollectionLabel = cms.InputTag("IsoProd","IsoTrackEcalRecHitCollection")
TrackAssociatorParameterBlock.TrackAssociatorParameters.HBHERecHitCollectionLabel = cms.InputTag("IsoProd","IsoTrackHBHERecHitCollection")
TrackAssociatorParameterBlock.TrackAssociatorParameters.HORecHitCollectionLabel = cms.InputTag("IsoProd","IsoTrackHORecHitCollection")

process.energyOverMomentumTree = cms.EDAnalyzer('EopTreeWriter',
    TrackAssociatorParameterBlock
)
process.energyOverMomentumTree.src = cms.InputTag('TrackRefitter')
#process.energyOverMomentumTree.src = cms.InputTag('TkAlIsoProd:')

process.TFileService = cms.Service("TFileService",
    fileName = cms.string('EopTree.root')
)

process.p = cms.Path(process.offlineBeamSpot*process.TrackRefitter*process.energyOverMomentumTree)
#process.p = cms.Path(process.offlineBeamSpot*process.AlignmentTrackSelector*process.TrackRefitter*process.energyOverMomentumTree)
