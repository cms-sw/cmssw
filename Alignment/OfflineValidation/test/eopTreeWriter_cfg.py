import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing
from Alignment.OfflineValidation.TkAlAllInOneTool.defaultInputFiles_cff import filesDefaultData_JetHTRun2018DHcalIsoTrk

options = VarParsing.VarParsing("analysis")

options.register ('GlobalTag',
                  'auto:run2_data',
                  VarParsing.VarParsing.multiplicity.singleton,  # singleton or list
                  VarParsing.VarParsing.varType.string,          # string, int, or float
                  "Global Tag to be used")

options.register('unitTest',
                 False, # default value
                 VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                 VarParsing.VarParsing.varType.bool, # string, int, or float
                 "is it a unit test?")
options.parseArguments()

process = cms.Process("EnergyOverMomentumTree")

####################################################################
# initialize MessageLogger and output report
####################################################################
process.load("FWCore.MessageService.MessageLogger_cfi")
#process.MessageLogger.cerr.threshold = 'ERROR'
process.MessageLogger.cerr.FwkReport.reportEvery = 100
process.MessageLogger.TrackRefitter=dict()

process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(False) )

# define input files
process.source = cms.Source("PoolSource", fileNames = filesDefaultData_JetHTRun2018DHcalIsoTrk)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(options.maxEvents)
)

print( "conditionGT       : ", options.GlobalTag)
print( "maxEvents         : ", options.maxEvents)

####################################################################
# load configuration files
####################################################################
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.Geometry.GeometryRecoDB_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, options.GlobalTag, '')

jsonFile = '/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions18/13TeV/Legacy_2018/Cert_314472-325175_13TeV_Legacy2018_Collisions18_JSON.txt'

if(options.unitTest):
    print('This is a unit test, will not json filter')
    pass
else:
    import FWCore.PythonUtilities.LumiList as LumiList
    import FWCore.ParameterSet.Types as CfgTypes
    print("JSON used: %s " % jsonFile)
    myLumis = LumiList.LumiList(filename = jsonFile).getCMSSWString().split(',')
    process.source.lumisToProcess = CfgTypes.untracked(CfgTypes.VLuminosityBlockRange())
    process.source.lumisToProcess.extend(myLumis)

#from TrackingTools.TrackAssociator.default_cfi import *
process.load("TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagator_cfi")
process.load("TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAny_cfi")
process.load("TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAlong_cfi")
process.load("TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorOpposite_cfi")
process.load("TrackingTools.TrackAssociator.DetIdAssociatorESProducer_cff")
from TrackingTools.TrackAssociator.default_cfi import *
#process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_noesprefer_cff")

####################################################################
# choose geometry
####################################################################
from CondCore.CondDB.CondDB_cfi import CondDB
CondDB.connect = cms.string("frontier://FrontierProd/CMS_CONDITIONS")

process.trackerAlignment = cms.ESSource("PoolDBESSource",
					CondDB,
                                        toGet = cms.VPSet(cms.PSet(record = cms.string("TrackerAlignmentRcd"),
                                                                   tag = cms.string("TrackerAlignment_v28_offline")
                                                               )))
process.es_prefer_trackerAlignment = cms.ESPrefer("PoolDBESSource", "trackerAlignment")

process.trackerAPE = cms.ESSource("PoolDBESSource",
                                  CondDB,
                                  toGet = cms.VPSet(cms.PSet(record = cms.string("TrackerAlignmentErrorRcd"),
                                                                 tag = cms.string("TrackerAlignmentExtendedErrors_v15_offline_IOVs")
                                                         )))
#process.es_prefer_TrackerAPE = cms.ESPrefer("PoolDBESSource", "trackerAPE")

process.load("RecoVertex.BeamSpotProducer.BeamSpot_cfi")

####################################################################
# configure alignment track selector
####################################################################
process.load("Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi")
process.AlignmentTrackSelector.src = cms.InputTag('TkAlIsoProdFilter') # 'TkAlIsoProd' # trackCollection' # 'ALCARECOTkAlZMuMu' # 'ALCARECOTkAlMinBias' # adjust to input file
process.AlignmentTrackSelector.ptMin = 1.
process.AlignmentTrackSelector.etaMin = -5.
process.AlignmentTrackSelector.etaMax = 5.
process.AlignmentTrackSelector.nHitMin = 5
process.AlignmentTrackSelector.chi2nMax = 100.
#process.AlignmentTrackSelector.applyNHighestPt = True
#process.AlignmentTrackSelector.nHighestPt = 2

####################################################################
# configure track refitter
####################################################################
process.load("RecoTracker.MeasurementDet.MeasurementTrackerEventProducer_cfi")
process.MeasurementTrackerEvent.pixelClusterProducer = "TkAlIsoProdFilter"
process.MeasurementTrackerEvent.stripClusterProducer = "TkAlIsoProdFilter"
#process.MeasurementTrackerEvent.inactivePixelDetectorLabels = cms.VInputTag([''])
#process.MeasurementTrackerEvent.inactiveStripDetectorLabels = cms.VInputTag([''])

import RecoTracker.TrackProducer.TrackRefitters_cff
process.load("RecoTracker.TrackProducer.TrackRefitters_cff")
process.TrackRefitter.src = cms.InputTag('AlignmentTrackSelector')
process.TrackRefitter.TrajectoryInEvent = True
process.TrackRefitter.NavigationSchool = ''

####################################################################
# configure tree writer
####################################################################

# uncomment following block in case it is run over the HcalCalIsoTrk AlCaReco
# TrackAssociatorParameterBlock.TrackAssociatorParameters.EBRecHitCollectionLabel = cms.InputTag("IsoProd", "IsoTrackEcalRecHitCollection")
# TrackAssociatorParameterBlock.TrackAssociatorParameters.EERecHitCollectionLabel = cms.InputTag("IsoProd", "IsoTrackEcalRecHitCollection")
# TrackAssociatorParameterBlock.TrackAssociatorParameters.HBHERecHitCollectionLabel = cms.InputTag("IsoProd", "IsoTrackHBHERecHitCollection")
# TrackAssociatorParameterBlock.TrackAssociatorParameters.HORecHitCollectionLabel = cms.InputTag("IsoProd", "IsoTrackHORecHitCollection")

TrackAssociatorParameterBlock.TrackAssociatorParameters.EERecHitCollectionLabel = cms.InputTag("ecalRecHit","EcalRecHitsEE")
TrackAssociatorParameterBlock.TrackAssociatorParameters.EBRecHitCollectionLabel = cms.InputTag("ecalRecHit","EcalRecHitsEB")
TrackAssociatorParameterBlock.TrackAssociatorParameters.HBHERecHitCollectionLabel = cms.InputTag("hbhereco")
TrackAssociatorParameterBlock.TrackAssociatorParameters.useHO = cms.bool(False)  # no HO hits saved in the alcareco

process.energyOverMomentumTree = cms.EDAnalyzer('EopTreeWriter',
                                                TrackAssociatorParameterBlock,
                                                src = cms.InputTag('TrackRefitter'))

####################################################################
# output file
####################################################################
process.TFileService = cms.Service("TFileService",
    fileName = cms.string('test_EopTree.root')
)

####################################################################
# Path
####################################################################
process.p = cms.Path(process.MeasurementTrackerEvent*
                     process.offlineBeamSpot*
                     process.AlignmentTrackSelector*
                     process.TrackRefitter*
                     process.energyOverMomentumTree)
