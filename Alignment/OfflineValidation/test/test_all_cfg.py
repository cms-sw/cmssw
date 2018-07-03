import FWCore.ParameterSet.Config as cms
import sys
from enum import Enum
from PhysicsTools.PatAlgos.patInputFiles_cff import filesRelValTTbarPileUpGENSIMRECO

class RefitType(Enum):
     STANDARD = 1
     COMMON   = 2
 
isDA = True
isMC = True
allFromGT = True
applyBows = True
applyExtraConditions = True
theRefitter = RefitType.COMMON

process = cms.Process("Demo") 

###################################################################
# Event source and run selection
###################################################################
process.source = cms.Source("PoolSource",
                            fileNames = filesRelValTTbarPileUpGENSIMRECO,
                            duplicateCheckMode = cms.untracked.string('checkAllFilesOpened')
                            )

runboundary = 1
process.source.firstRun = cms.untracked.uint32(int(runboundary))
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(10) )

###################################################################
# JSON Filtering
###################################################################
if isMC:
     print ">>>>>>>>>> testPVValidation_cfg.py: msg%-i: This is Simulation!"
     runboundary = 1
else:
     print ">>>>>>>>>> testPVValidation_cfg.py: msg%-i: This is DATA!"
     import FWCore.PythonUtilities.LumiList as LumiList
     process.source.lumisToProcess = LumiList.LumiList(filename ='None').getVLuminosityBlockRange()

###################################################################
# Messages
###################################################################
process.load('FWCore.MessageService.MessageLogger_cfi')   
process.MessageLogger.categories.append("PrimaryVertexValidation")  
process.MessageLogger.categories.append("FilterOutLowPt")  
process.MessageLogger.destinations = cms.untracked.vstring("cout")
process.MessageLogger.cout = cms.untracked.PSet(
    threshold = cms.untracked.string("INFO"),
    default   = cms.untracked.PSet(limit = cms.untracked.int32(0)),                       
    FwkReport = cms.untracked.PSet(limit = cms.untracked.int32(-1),
                                   reportEvery = cms.untracked.int32(1000)
                                   ),                                                      
    PrimaryVertexValidation = cms.untracked.PSet( limit = cms.untracked.int32(-1)),
    FilterOutLowPt          = cms.untracked.PSet( limit = cms.untracked.int32(-1))
    )
process.MessageLogger.statistics.append('cout') 

####################################################################
# Produce the Transient Track Record in the event
####################################################################
process.load("TrackingTools.TransientTrack.TransientTrackBuilder_cfi")

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
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase1_2017_realistic', '')

if allFromGT:
     print ">>>>>>>>>> testPVValidation_cfg.py: msg%-i: All is taken from GT"
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
     if applyBows:
          print ">>>>>>>>>> testPVValidation_cfg.py: msg%-i: Applying TrackerSurfaceDeformations!"
          process.trackerBows = cms.ESSource("PoolDBESSource",CondDBSetup,
                                             connect = cms.string('frontier://FrontierProd/CMS_CONDITIONS'),
                                             toGet = cms.VPSet(cms.PSet(record = cms.string('TrackerSurfaceDeformationRcd'),
                                                                        tag = cms.string('TrackerSurfaceDeformations_zero')
                                                                        )
                                                               )
                                             )
          process.es_prefer_Bows = cms.ESPrefer("PoolDBESSource", "trackerBows")
     else:
          print ">>>>>>>>>> testPVValidation_cfg.py: msg%-i: MultiPVValidation: Not applying TrackerSurfaceDeformations!"

     ####################################################################
     # Extra corrections not included in the GT
     ####################################################################
     if applyExtraConditions:

          import CalibTracker.Configuration.Common.PoolDBESSource_cfi
          ##### END OF EXTRA CONDITIONS
 
     else:
          print ">>>>>>>>>> testPVValidation_cfg.py: msg%-i: Not applying extra calibration constants!"
     
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
                                  src =  cms.untracked.InputTag("generalTracks"),
                                  debugOn = cms.untracked.bool(False),
                                  numtrack = cms.untracked.uint32(10),
                                  thresh = cms.untracked.double(0.25)
                                  )

process.noslowpt = cms.EDFilter("FilterOutLowPt",
                                applyfilter = cms.untracked.bool(True),
                                src =  cms.untracked.InputTag("generalTracks"),
                                debugOn = cms.untracked.bool(False),
                                numtrack = cms.untracked.uint32(0),
                                thresh = cms.untracked.int32(1),
                                ptmin  = cms.untracked.double(3.),
                                runControl = cms.untracked.bool(True),
                                runControlNumber = cms.untracked.vuint32(int(runboundary))
                                )

if isMC:
     process.goodvertexSkim = cms.Sequence(process.noscraping)
else:
     process.goodvertexSkim = cms.Sequence(process.primaryVertexFilter + process.noscraping + process.noslowpt)


if(theRefitter == RefitType.COMMON):

     print ">>>>>>>>>> testPVValidation_cfg.py: msg%-i: using the common track selection and refit sequence!"          
     ####################################################################
     # Load and Configure Common Track Selection and refitting sequence
     ####################################################################
     import Alignment.CommonAlignment.tools.trackselectionRefitting as trackselRefit
     process.seqTrackselRefit = trackselRefit.getSequence(process, 'generalTracks',
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
     
elif (theRefitter == RefitType.STANDARD):

     print ">>>>>>>>>> testPVValidation_cfg.py: msg%-i: using the standard single refit sequence!"          
     ####################################################################
     # Load and Configure Measurement Tracker Event
     # (needed in case NavigationSchool is set != '')
     ####################################################################
     # process.load("RecoTracker.MeasurementDet.MeasurementTrackerEventProducer_cfi") 
     # process.MeasurementTrackerEvent.pixelClusterProducer = 'generalTracks'
     # process.MeasurementTrackerEvent.stripClusterProducer = 'generalTracks'
     # process.MeasurementTrackerEvent.inactivePixelDetectorLabels = cms.VInputTag()
     # process.MeasurementTrackerEvent.inactiveStripDetectorLabels = cms.VInputTag()

     ####################################################################
     # Load and Configure TrackRefitter
     ####################################################################
     process.load("RecoTracker.TrackProducer.TrackRefitters_cff")
     import RecoTracker.TrackProducer.TrackRefitters_cff
     process.FinalTrackRefitter = RecoTracker.TrackProducer.TrackRefitter_cfi.TrackRefitter.clone()
     process.FinalTrackRefitter.src = "generalTracks"
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
# Deterministic annealing clustering
####################################################################
if isDA:
     print ">>>>>>>>>> testPVValidation_cfg.py: msg%-i: Running DA Algorithm!"
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
                                           
                                           TkFilterParameters = cms.PSet(algorithm=cms.string('filter'),                           
                                                                         maxNormalizedChi2 = cms.double(5.0),                        # chi2ndof < 5                  
                                                                         minPixelLayersWithHits = cms.int32(2),                      # PX hits > 2                       
                                                                         minSiliconLayersWithHits = cms.int32(5),                    # TK hits > 5  
                                                                         maxD0Significance = cms.double(5.0),                        # fake cut (requiring 1 PXB hit)     
                                                                         minPt = cms.double(0.0),                                    # better for softish events
                                                                         maxEta = cms.double(5.0),                                   # as per recommendation in PR #18330
                                                                         trackQuality = cms.string("any")
                                                                         ),
                                           
                                           ## MM 04.05.2017 (use settings as in: https://github.com/cms-sw/cmssw/pull/18330)
                                           TkClusParameters=cms.PSet(algorithm=cms.string('DA_vect'),
                                                                     TkDAClusParameters = cms.PSet(coolingFactor = cms.double(0.6),  # moderate annealing speed
                                                                                                   Tmin = cms.double(2.0),           # end of vertex splitting
                                                                                                   Tpurge = cms.double(2.0),         # cleaning 
                                                                                                   Tstop = cms.double(0.5),          # end of annealing
                                                                                                   vertexSize = cms.double(0.006),   # added in quadrature to track-z resolutions
                                                                                                   d0CutOff = cms.double(3.),        # downweight high IP tracks
                                                                                                   dzCutOff = cms.double(3.),        # outlier rejection after freeze-out (T<Tmin)   
                                                                                                   zmerge = cms.double(1e-2),        # merge intermediat clusters separated by less than zmerge
                                                                                                   uniquetrkweight = cms.double(0.8) # require at least two tracks with this weight at T=Tpurge
                                                                                                   )
                                                                     )
                                           )

####################################################################
# GAP clustering
####################################################################
else:
     print ">>>>>>>>>> testPVValidation_cfg.py: msg%-i: Running GAP Algorithm!"
     process.PVValidation = cms.EDAnalyzer("PrimaryVertexValidation",
                                           TrackCollectionTag = cms.InputTag("FinalTrackRefitter"),
                                           VertexCollectionTag = cms.InputTag("offlinePrimaryVertices"), 
                                           Debug = cms.bool(False),
                                           isLightNtuple = cms.bool(True),
                                           storeNtuple = cms.bool(False),
                                           useTracksFromRecoVtx = cms.bool(False),
                                           askFirstLayerHit = cms.bool(False),
                                           forceBeamSpot = cms.untracked.bool(False),
                                           probePt = cms.untracked.double(3.),
                                           minPt   = cms.untracked.double(1.),
                                           maxPt   = cms.untracked.double(30.),
                                           runControl = cms.untracked.bool(True),
                                           runControlNumber = cms.untracked.vuint32(int(runboundary)),
                                           
                                           TkFilterParameters = cms.PSet(algorithm=cms.string('filter'),                             
                                                                         maxNormalizedChi2 = cms.double(5.0),                        # chi2ndof < 20                  
                                                                         minPixelLayersWithHits=cms.int32(2),                        # PX hits > 2                   
                                                                         minSiliconLayersWithHits = cms.int32(5),                    # TK hits > 5                   
                                                                         maxD0Significance = cms.double(5.0),                        # fake cut (requiring 1 PXB hit)
                                                                         minPt = cms.double(0.0),                                    # better for softish events    
                                                                         maxEta = cms.double(5.0),                                   # as per recommendation in PR #18330 
                                                                         trackQuality = cms.string("any")
                                                                         ),
                                        
                                           TkClusParameters = cms.PSet(algorithm   = cms.string('gap'),
                                                                       TkGapClusParameters = cms.PSet(zSeparation = cms.double(0.2)  # 0.2 cm max separation betw. clusters
                                                                                                      ) 
                                                                       )
                                           )



####################################################################
# Path
####################################################################
process.p = cms.Path(process.goodvertexSkim*
                     process.seqTrackselRefit*
                     process.PVValidation)
