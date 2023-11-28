import sys
import FWCore.ParameterSet.Config as cms

import FWCore.ParameterSet.VarParsing as VarParsing
options = VarParsing.VarParsing()
options.register("isPhase2",
                 False,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.bool,
                 "is the test running with phase-2 geometry")
options.register("maxEvents",
                 -1,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "number of events to run")
options.parseArguments()

if (options.isPhase2):
     from Alignment.OfflineValidation.TkAlAllInOneTool.defaultInputFiles_cff import filesDefaultMC_TTbarPhase2RECO
else:
     from Alignment.OfflineValidation.TkAlAllInOneTool.defaultInputFiles_cff import filesDefaultMC_TTBarPU

from enum import Enum
class RefitType(Enum):
     STANDARD = 1
     COMMON   = 2
 
_isDA = True
_isMC = True
_allFromGT = True
_applyBows = True
_applyExtraConditions = True
_theRefitter = RefitType.COMMON # RefitType.STANDARD (other option not involving filtering)
_theTrackCollection = 'generalTracks' # FIXME: 'ALCARECOTkAlMinBias' once a sample is available

###################################################################
# Set the era
###################################################################
if(options.isPhase2):
     from Configuration.Eras.Era_Phase2C17I13M9_cff import Phase2C17I13M9
     process = cms.Process("Demo",Phase2C17I13M9) 
else:
     from Configuration.Eras.Era_Run3_cff import Run3
     process = cms.Process("Demo", Run3)
     
###################################################################
# Set the process to run multi-threaded
###################################################################
process.options.numberOfThreads = 8

###################################################################
# Event source and run selection
###################################################################
process.source = cms.Source("PoolSource",
                            fileNames = (filesDefaultMC_TTbarPhase2RECO if (options.isPhase2) else filesDefaultMC_TTBarPU),
                            duplicateCheckMode = cms.untracked.string('checkAllFilesOpened'))

runboundary = 1
process.source.firstRun = cms.untracked.uint32(int(runboundary))
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(options.maxEvents))  #(10 if (options.isPhase2) else 100)

###################################################################
# JSON Filtering
###################################################################
if _isMC:
     print("############ testPVValidation_cfg.py: msg%-i: This is Simulation!")
     runboundary = 1
else:
     print("############ testPVValidation_cfg.py: msg%-i: This is DATA!")
     import FWCore.PythonUtilities.LumiList as LumiList
     process.source.lumisToProcess = LumiList.LumiList(filename ='None').getVLuminosityBlockRange()

###################################################################
# Messages
###################################################################
process.load('FWCore.MessageService.MessageLogger_cfi')   
process.MessageLogger.cerr.enable = False
process.MessageLogger.PrimaryVertexValidation=dict()  
process.MessageLogger.SplitVertexResolution=dict()
process.MessageLogger.FilterOutLowPt=dict()  
process.MessageLogger.cout = cms.untracked.PSet(
    enable = cms.untracked.bool(True),
    threshold = cms.untracked.string("INFO"),
    default   = cms.untracked.PSet(limit = cms.untracked.int32(0)),                       
    FwkReport = cms.untracked.PSet(limit = cms.untracked.int32(-1),
                                   reportEvery = cms.untracked.int32(1 if (options.isPhase2) else 10)),                                                      
    PrimaryVertexValidation = cms.untracked.PSet( limit = cms.untracked.int32(-1)),
    SplitVertexResolution   = cms.untracked.PSet( limit = cms.untracked.int32(-1)),
    FilterOutLowPt          = cms.untracked.PSet( limit = cms.untracked.int32(-1)),
    enableStatistics = cms.untracked.bool(True)
    )

####################################################################
# Produce the Transient Track Record in the event
####################################################################
process.load("TrackingTools.TransientTrack.TransientTrackBuilder_cfi")

####################################################################
# Get the Magnetic Field
####################################################################
process.load('Configuration.StandardSequences.MagneticField_cff')

###################################################################
# Standard loads
###################################################################
if(options.isPhase2):
     process.load('Configuration.Geometry.GeometryExtended2026D88Reco_cff')
else:
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
process.GlobalTag = GlobalTag(process.GlobalTag, ('auto:phase2_realistic_T21' if options.isPhase2 else 'auto:phase1_2022_realistic'), '')

if _allFromGT:
     print("############ testPVValidation_cfg.py: msg%-i: All is taken from GT")
else:
     if(options.isPhase2):
          print("########## overriding of phase-2 alignment conditions is not yet supported")
          pass
     else:
          ####################################################################
          # Get Alignment constants
          ####################################################################
          from CondCore.DBCommon.CondDBSetup_cfi import *
          process.trackerAlignment = cms.ESSource("PoolDBESSource",CondDBSetup,
                                                  connect = cms.string('frontier://FrontierProd/CMS_CONDITIONS'),
                                                  timetype = cms.string("runnumber"),
                                                  toGet = cms.VPSet(cms.PSet(record = cms.string('TrackerAlignmentRcd'),
                                                                             tag = cms.string('TrackerAlignment_Upgrade2017_design_v4')
                                                                        )
                                                               )
                                             )
          process.es_prefer_trackerAlignment = cms.ESPrefer("PoolDBESSource", "trackerAlignment")

          ####################################################################
          # Get APE
          ####################################################################
          process.setAPE = cms.ESSource("PoolDBESSource",CondDBSetup,
                                        connect = cms.string('frontier://FrontierProd/CMS_CONDITIONS'),
                                        timetype = cms.string("runnumber"),
                                        toGet = cms.VPSet(cms.PSet(record = cms.string('TrackerAlignmentErrorExtendedRcd'),
                                                                   tag = cms.string('TrackerAlignmentErrorsExtended_Upgrade2017_design_v0')
                                                              )
                                                     )
                                   )
          process.es_prefer_setAPE = cms.ESPrefer("PoolDBESSource", "setAPE")

          ####################################################################
          # Kinks and Bows (optional)
          ####################################################################
          if _applyBows:
               print("############ testPVValidation_cfg.py: msg%-i: Applying TrackerSurfaceDeformations!")
               process.trackerBows = cms.ESSource("PoolDBESSource",CondDBSetup,
                                                  connect = cms.string('frontier://FrontierProd/CMS_CONDITIONS'),
                                                  toGet = cms.VPSet(cms.PSet(record = cms.string('TrackerSurfaceDeformationRcd'),
                                                                             tag = cms.string('TrackerSurfaceDeformations_zero')
                                                                        )
                                                               )
                                             )
               process.es_prefer_Bows = cms.ESPrefer("PoolDBESSource", "trackerBows")
          else:
               print("############ testPVValidation_cfg.py: msg%-i: MultiPVValidation: Not applying TrackerSurfaceDeformations!")

     ####################################################################
     # Extra corrections not included in the GT
     ####################################################################
     if _applyExtraConditions:

          import CalibTracker.Configuration.Common.PoolDBESSource_cfi
          ##### END OF EXTRA CONDITIONS
 
     else:
          print("############ testPVValidation_cfg.py: msg%-i: Not applying extra calibration constants!")
     
####################################################################
# Load and Configure event selection
####################################################################
process.primaryVertexFilter = cms.EDFilter("VertexSelector",
                                             src = cms.InputTag("offlinePrimaryVertices"),
                                             cut = cms.string("!isFake && ndof > 4 && abs(z) <= 24 && position.Rho <= 2"),
                                             filter = cms.bool(True)
                                             )

process.noscraping = cms.EDFilter("FilterOutScraping",
                                  applyfilter = cms.untracked.bool(True),
                                  src =  cms.untracked.InputTag(_theTrackCollection),
                                  debugOn = cms.untracked.bool(False),
                                  numtrack = cms.untracked.uint32(10),
                                  thresh = cms.untracked.double(0.25)
                                  )

process.noslowpt = cms.EDFilter("FilterOutLowPt",
                                applyfilter = cms.untracked.bool(True),
                                src =  cms.untracked.InputTag(_theTrackCollection),
                                debugOn = cms.untracked.bool(False),
                                numtrack = cms.untracked.uint32(0),
                                thresh = cms.untracked.int32(1),
                                ptmin  = cms.untracked.double(3.),
                                runControl = cms.untracked.bool(True),
                                runControlNumber = cms.untracked.vuint32(int(runboundary))
                                )

if _isMC:
     process.goodvertexSkim = cms.Sequence(process.noscraping)
else:
     process.goodvertexSkim = cms.Sequence(process.primaryVertexFilter + process.noscraping + process.noslowpt)


if(_theRefitter == RefitType.COMMON):

     print("############ testPVValidation_cfg.py: msg%-i: using the common track selection and refit sequence!")
     ####################################################################
     # Load and Configure Common Track Selection and refitting sequence
     ####################################################################
     import Alignment.CommonAlignment.tools.trackselectionRefitting as trackselRefit
     process.seqTrackselRefit = trackselRefit.getSequence(process, _theTrackCollection,
                                                          isPVValidation=True, 
                                                          TTRHBuilder='WithAngleAndTemplate',
                                                          usePixelQualityFlag=True,
                                                          openMassWindow=False,
                                                          cosmicsDecoMode=True,
                                                          cosmicsZeroTesla=False,
                                                          momentumConstraint=None,
                                                          cosmicTrackSplitting=False,
                                                          use_d0cut=False,
                                                          )
     
elif (_theRefitter == RefitType.STANDARD):

     print("############ testPVValidation_cfg.py: msg%-i: using the standard single refit sequence!")
     ####################################################################
     # Load and Configure Measurement Tracker Event
     # (needed in case NavigationSchool is set != '')
     ####################################################################
     # process.load("RecoTracker.MeasurementDet.MeasurementTrackerEventProducer_cfi") 
     # process.MeasurementTrackerEvent.pixelClusterProducer = '_theTrackCollection'
     # process.MeasurementTrackerEvent.stripClusterProducer = '_theTrackCollection'
     # process.MeasurementTrackerEvent.inactivePixelDetectorLabels = cms.VInputTag()
     # process.MeasurementTrackerEvent.inactiveStripDetectorLabels = cms.VInputTag()

     ####################################################################
     # Load and Configure TrackRefitter
     ####################################################################
     process.load("RecoTracker.TrackProducer.TrackRefitters_cff")
     import RecoTracker.TrackProducer.TrackRefitters_cff
     process.FinalTrackRefitter = RecoTracker.TrackProducer.TrackRefitter_cfi.TrackRefitter.clone()
     process.FinalTrackRefitter.src = _theTrackCollection
     process.FinalTrackRefitter.TrajectoryInEvent = True
     process.FinalTrackRefitter.NavigationSchool = ''
     process.FinalTrackRefitter.TTRHBuilder = "WithAngleAndTemplate"

     ####################################################################
     # Sequence
     ####################################################################
     process.seqTrackselRefit = cms.Sequence(process.offlineBeamSpot*
                                             # in case NavigatioSchool is set !='' 
                                             #process.MeasurementTrackerEvent*
                                             process.FinalTrackRefitter)     

####################################################################
# Output file
####################################################################
process.TFileService = cms.Service("TFileService",
                                   fileName=cms.string("PVValidation_test_0.root")
                                  )

####################################################################
# Imports of parameters
####################################################################
from RecoVertex.PrimaryVertexProducer.OfflinePrimaryVertices_cfi import offlinePrimaryVertices
## modify the parameters which differ
FilteringParams = offlinePrimaryVertices.TkFilterParameters.clone(
     maxNormalizedChi2 = 5.0,  # chi2ndof < 5
     maxD0Significance = 5.0,  # fake cut (requiring 1 PXB hit)
     maxEta = 5.0,             # as per recommendation in PR #18330
)

## MM 04.05.2017 (use settings as in: https://github.com/cms-sw/cmssw/pull/18330)
from RecoVertex.PrimaryVertexProducer.TkClusParameters_cff import DA_vectParameters
DAClusterizationParams = DA_vectParameters.clone()

GapClusterizationParams = cms.PSet(algorithm   = cms.string('gap'),
                                   TkGapClusParameters = cms.PSet(zSeparation = cms.double(0.2))  # 0.2 cm max separation betw. clusters
                                   )

####################################################################
# Deterministic annealing clustering or Gap clustering
####################################################################
def switchClusterizerParameters(da):
     if da:
          print("############ testPVValidation_cfg.py: msg%-i: Running DA Algorithm!")
          return DAClusterizationParams
     else:
          print("############ testPVValidation_cfg.py: msg%-i: Running GAP Algorithm!")
          return GapClusterizationParams

####################################################################
# Print table of execution parameters
####################################################################
from prettytable import PrettyTable
x = PrettyTable()
x.field_names = ["Parameter","Value"]
x.add_row(["is DA",_isDA])
x.add_row(["is MC",_isMC])
x.add_row(["applyBows",_applyBows])
x.add_row(["all from GT",_allFromGT])
x.add_row(["extra conditions",_applyExtraConditions])
x.add_row(["refitter type",_theRefitter])
x.add_row(["track collection",_theTrackCollection])
x.add_row(["GlobalTag",process.GlobalTag.globaltag.value()])
x.add_row(["# events",options.maxEvents])
print(x)

####################################################################
# Configure the PVValidation Analyzer module
####################################################################
process.PVValidation = cms.EDAnalyzer("PrimaryVertexValidation",
                                      TrackCollectionTag = cms.InputTag("FinalTrackRefitter"),
                                      VertexCollectionTag = cms.InputTag("offlinePrimaryVertices"),
                                      Debug = cms.bool(False),
                                      storeNtuple = cms.bool(False),
                                      useTracksFromRecoVtx = cms.bool(False),
                                      isLightNtuple = cms.bool(True),
                                      askFirstLayerHit = cms.bool(False),
                                      forceBeamSpot = cms.untracked.bool(False),
                                      probePt = cms.untracked.double(3.),
                                      minPt   = cms.untracked.double(1.),
                                      maxPt   = cms.untracked.double(30.),
                                      runControl = cms.untracked.bool(True),
                                      runControlNumber = cms.untracked.vuint32(int(runboundary)),
                                      TkFilterParameters = FilteringParams,
                                      TkClusParameters = switchClusterizerParameters(_isDA)
                                      )

####################################################################
# Path
####################################################################
process.p = cms.Path(process.goodvertexSkim*
                     process.seqTrackselRefit*
                     process.PVValidation)

## PV refit part
process.load("TrackingTools.TransientTrack.TransientTrackBuilder_cfi")
process.offlinePrimaryVerticesFromRefittedTrks  = offlinePrimaryVertices.clone()
process.offlinePrimaryVerticesFromRefittedTrks.TrackLabel                                       = cms.InputTag("FinalTrackRefitter")
process.offlinePrimaryVerticesFromRefittedTrks.vertexCollections.maxDistanceToBeam              = 1
process.offlinePrimaryVerticesFromRefittedTrks.TkFilterParameters.maxNormalizedChi2             = 20
process.offlinePrimaryVerticesFromRefittedTrks.TkFilterParameters.minSiliconLayersWithHits      = 5
process.offlinePrimaryVerticesFromRefittedTrks.TkFilterParameters.maxD0Significance             = 5.0
# as it was prior to https://github.com/cms-sw/cmssw/commit/c8462ae4313b6be3bbce36e45373aa6e87253c59
process.offlinePrimaryVerticesFromRefittedTrks.TkFilterParameters.maxD0Error                    = 1.0
process.offlinePrimaryVerticesFromRefittedTrks.TkFilterParameters.maxDzError                    = 1.0
process.offlinePrimaryVerticesFromRefittedTrks.TkFilterParameters.minPixelLayersWithHits        = 2

###################################################################
# The trigger filter module
###################################################################
from HLTrigger.HLTfilters.triggerResultsFilter_cfi import *
process.HLTFilter = triggerResultsFilter.clone(
     triggerConditions = cms.vstring("HLT_ZeroBias_*"),
     #triggerConditions = cms.vstring("HLT_HT*"),
     hltResults = cms.InputTag( "TriggerResults", "", "HLT" ),
     l1tResults = cms.InputTag( "" ),
     throw = cms.bool(False)
)

###################################################################
# The analysis module
###################################################################
process.trackanalysis = cms.EDAnalyzer("GeneralPurposeTrackAnalyzer",
                                       TkTag  = cms.InputTag('FinalTrackRefitter'),
                                       isCosmics = cms.bool(False))

process.vertexanalysis = cms.EDAnalyzer('GeneralPurposeVertexAnalyzer',
                                        ndof = cms.int32(4),
                                        vertexLabel = cms.InputTag('offlinePrimaryVerticesFromRefittedTrks'),
                                        beamSpotLabel = cms.InputTag('offlineBeamSpot'),
                                        Xpos = cms.double(0.1),
                                        Ypos = cms.double(0),
                                        TkSizeBin = cms.int32(100),
                                        TkSizeMin = cms.double(499.5),
                                        TkSizeMax = cms.double(-0.5),
                                        DxyBin = cms.int32(100),
                                        DxyMin = cms.double(5000),
                                        DxyMax = cms.double(-5000),
                                        DzBin = cms.int32(100),
                                        DzMin = cms.double(-2000),
                                        DzMax = cms.double(2000),
                                        PhiBin = cms.int32(32),
                                        PhiBin2D = cms.int32(12),
                                        PhiMin = cms.double(-3.1415926535897931),
                                        PhiMax = cms.double(3.1415926535897931),
                                        EtaBin = cms.int32(26),
                                        EtaBin2D = cms.int32(8),
                                        EtaMin = cms.double(-2.7),
                                        EtaMax = cms.double(2.7))

###################################################################
# The PV resolution module
###################################################################
process.PrimaryVertexResolution = cms.EDAnalyzer('SplitVertexResolution',
                                                 storeNtuple         = cms.bool(True),
                                                 vtxCollection       = cms.InputTag("offlinePrimaryVerticesFromRefittedTrks"),
                                                 trackCollection     = cms.InputTag("FinalTrackRefitter"),
                                                 minVertexNdf        = cms.untracked.double(10.),
                                                 minVertexMeanWeight = cms.untracked.double(0.5),
                                                 runControl = cms.untracked.bool(True),
                                                 runControlNumber = cms.untracked.vuint32(int(runboundary))
                                                 )

process.p2 = cms.Path(process.HLTFilter                               +
                      process.seqTrackselRefit                        +
                      process.offlinePrimaryVerticesFromRefittedTrks  +
                      process.PrimaryVertexResolution                 +
                      process.trackanalysis                           +
                      process.vertexanalysis
                      )
