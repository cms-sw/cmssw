import FWCore.ParameterSet.Config as cms
from enum import Enum
import sys
 
class RefitType(Enum):
     STANDARD = 1
     COMMON   = 2

isDA = ISDATEMPLATE
isMC = ISMCTEMPLATE
allFromGT = ALLFROMGTTEMPLATE
applyBows = APPLYBOWSTEMPLATE
applyExtraConditions = EXTRACONDTEMPLATE
theRefitter = REFITTERTEMPLATE

process = cms.Process("PrimaryVertexValidation") 

###################################################################
# Set the process to run multi-threaded
###################################################################
process.options.numberOfThreads = 8

###################################################################
def customiseAlignmentAndAPE(process):
###################################################################
    if not hasattr(process.GlobalTag,'toGet'):
        process.GlobalTag.toGet=cms.VPSet()
    process.GlobalTag.toGet.extend( cms.VPSet(cms.PSet(record = cms.string("TrackerAlignmentRcd"),
                                                       tag = cms.string("GEOMTAGTEMPLATE"),
                                                       connect = cms.string("ALIGNOBJTEMPLATE")
                                                       ),
                                              cms.PSet(record = cms.string("TrackerAlignmentErrorExtendedRcd"),
                                                       tag = cms.string("ERRORTAGTEMPLATE"),
                                                       connect = cms.string("APEOBJTEMPLATE")
                                                       )
                                              )
                                    )
    return process

###################################################################
def customiseKinksAndBows(process):
###################################################################
     if not hasattr(process.GlobalTag,'toGet'):
          process.GlobalTag.toGet=cms.VPSet()
     process.GlobalTag.toGet.extend(cms.VPSet(cms.PSet(record = cms.string("TrackerSurfaceDeformationRcd"),
                                                       tag = cms.string("BOWSTAGTEMPLATE"),
                                                       connect = cms.string("BOWSOBJECTTEMPLATE")
                                                       ),        
                                              )
                                    )
     return process

###################################################################
# Event source and run selection
###################################################################
readFiles = cms.untracked.vstring()
readFiles.extend(FILESOURCETEMPLATE)
process.source = cms.Source("PoolSource",
                            fileNames = readFiles ,
                            duplicateCheckMode = cms.untracked.string('checkAllFilesOpened')
                            )

runboundary = RUNBOUNDARYTEMPLATE
process.source.firstRun = cms.untracked.uint32(int(runboundary))
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(MAXEVENTSTEMPLATE) )

###################################################################
# JSON Filtering
###################################################################
if isMC:
     print("############ testPVValidation_cfg.py: msg%-i: This is Simulation!")
     runboundary = 1
else:
     print("############ testPVValidation_cfg.py: msg%-i: This is DATA!")
     import FWCore.PythonUtilities.LumiList as LumiList
     process.source.lumisToProcess = LumiList.LumiList(filename ='LUMILISTTEMPLATE').getVLuminosityBlockRange()

###################################################################
# Messages
###################################################################
process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 1000

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
process.GlobalTag = GlobalTag(process.GlobalTag, 'GLOBALTAGTEMPLATE', '')

if allFromGT:
     print("############ testPVValidation_cfg.py: msg%-i: All is taken from GT")
else:
     ####################################################################
     # Get Alignment constants and APE
     ####################################################################
     process=customiseAlignmentAndAPE(process)

     ####################################################################
     # Kinks and Bows (optional)
     ####################################################################
     if applyBows:
          print("############ testPVValidation_cfg.py: msg%-i: Applying TrackerSurfaceDeformations!")
          process=customiseKinksAndBows(process)
     else:
          print("############ testPVValidation_cfg.py: msg%-i: MultiPVValidation: Not applying TrackerSurfaceDeformations!")

     ####################################################################
     # Extra corrections not included in the GT
     ####################################################################
     if applyExtraConditions:

          import CalibTracker.Configuration.Common.PoolDBESSource_cfi
          ##### END OF EXTRA CONDITIONS
 
     else:
          print("############ testPVValidation_cfg.py: msg%-i: Not applying extra calibration constants!")
     
####################################################################
# Load and Configure event selection
####################################################################
process.primaryVertexFilter = cms.EDFilter("VertexSelector",
                                             src = cms.InputTag("VERTEXTYPETEMPLATE"),
                                             cut = cms.string("!isFake && ndof > 4 && abs(z) <= 24 && position.Rho <= 2"),
                                             filter = cms.bool(True)
                                             )

process.noscraping = cms.EDFilter("FilterOutScraping",
                                  applyfilter = cms.untracked.bool(True),
                                  src =  cms.untracked.InputTag("TRACKTYPETEMPLATE"),
                                  debugOn = cms.untracked.bool(False),
                                  numtrack = cms.untracked.uint32(10),
                                  thresh = cms.untracked.double(0.25)
                                  )

process.load("Alignment.CommonAlignment.filterOutLowPt_cfi")
process.filterOutLowPt.applyfilter = True
process.filterOutLowPt.src = "TRACKTYPETEMPLATE"
process.filterOutLowPt.numtrack = 0
process.filterOutLowPt.thresh = 1
process.filterOutLowPt.ptmin  = PTCUTTEMPLATE
process.filterOutLowPt.runControl = RUNCONTROLTEMPLATE
process.filterOutLowPt.runControlNumber = [runboundary]

if isMC:
     process.goodvertexSkim = cms.Sequence(process.noscraping+process.filterOutLowPt)
else:
     process.goodvertexSkim = cms.Sequence(process.primaryVertexFilter + process.noscraping + process.filterOutLowPt)

####################################################################
# Load and Configure Measurement Tracker Event
# (this would be needed in case NavigationSchool is set != from ''
####################################################################
# process.load("RecoTracker.MeasurementDet.MeasurementTrackerEventProducer_cfi") 
# process.MeasurementTrackerEvent.pixelClusterProducer = 'TRACKTYPETEMPLATE'
# process.MeasurementTrackerEvent.stripClusterProducer = 'TRACKTYPETEMPLATE'
# process.MeasurementTrackerEvent.inactivePixelDetectorLabels = cms.VInputTag()
# process.MeasurementTrackerEvent.inactiveStripDetectorLabels = cms.VInputTag()

####################################################################
# Load and Configure TrackRefitter
####################################################################
# process.load("RecoTracker.TrackProducer.TrackRefitters_cff")
# import RecoTracker.TrackProducer.TrackRefitters_cff
# process.FinalTrackRefitter = RecoTracker.TrackProducer.TrackRefitter_cfi.TrackRefitter.clone()
# process.FinalTrackRefitter.src = "TRACKTYPETEMPLATE"
# process.FinalTrackRefitter.TrajectoryInEvent = True
# process.FinalTrackRefitter.NavigationSchool = ''
# process.FinalTrackRefitter.TTRHBuilder = "WithAngleAndTemplate"

####################################################################
# Load and Configure common selection sequence
####################################################################
# import Alignment.CommonAlignment.tools.trackselectionRefitting as trackselRx1efit
# process.seqTrackselRefit = trackselRefit.getSequence(process,'TRACKTYPETEMPLATE')
# process.HighPurityTrackSelector.trackQualities = cms.vstring()
# process.HighPurityTrackSelector.pMin     = cms.double(0.)
# #process.TrackerTrackHitFilter.usePixelQualityFlag = cms.bool(False)    # do not use the pixel quality flag
# #process.TrackerTrackHitFilter.commands   = cms.vstring("drop PXB 1")   # drop BPix1 hits
# process.AlignmentTrackSelector.pMin      = cms.double(0.)
# process.AlignmentTrackSelector.ptMin     = cms.double(0.)
# process.AlignmentTrackSelector.nHitMin2D = cms.uint32(0)
# process.AlignmentTrackSelector.nHitMin   = cms.double(0.)
# process.AlignmentTrackSelector.d0Min     = cms.double(-999999.0)
# process.AlignmentTrackSelector.d0Max     = cms.double(+999999.0)
# process.AlignmentTrackSelector.dzMin     = cms.double(-999999.0)
# process.AlignmentTrackSelector.dzMax     = cms.double(+999999.0)

if(theRefitter == RefitType.COMMON):

     print("############ testPVValidation_cfg.py: msg%-i: using the common track selection and refit sequence!")
     ####################################################################
     # Load and Configure Common Track Selection and refitting sequence
     ####################################################################
     import Alignment.CommonAlignment.tools.trackselectionRefitting as trackselRefit
     process.seqTrackselRefit = trackselRefit.getSequence(process, 'TRACKTYPETEMPLATE',
                                                          isPVValidation=True,
                                                          TTRHBuilder='TTRHBUILDERTEMPLATE',
                                                          usePixelQualityFlag=True,
                                                          openMassWindow=False,
                                                          cosmicsDecoMode=True,
                                                          cosmicsZeroTesla=False,
                                                          momentumConstraint=None,
                                                          cosmicTrackSplitting=False,
                                                          use_d0cut=False,
                                                          )
     if((process.TrackerTrackHitFilter.usePixelQualityFlag.value()==True) and (process.FirstTrackRefitter.TTRHBuilder.value()=="WithTrackAngle")):
          print(" \n\n","*"*70,"\n *\t\t\t\t WARNING!!!!!\t\t\t\n *\n * Found an inconsistent configuration!\n * TTRHBuilder = WithTrackAngle requires usePixelQualityFlag = False.\n * Going to reset it! \n *\n","*"*70)
          process.TrackerTrackHitFilter.usePixelQualityFlag = False

elif (theRefitter == RefitType.STANDARD):

     print("############ testPVValidation_cfg.py: msg%-i: using the standard single refit sequence!")
     ####################################################################
     # Load and Configure Measurement Tracker Event
     # (needed in case NavigationSchool is set != '')
     ####################################################################
     # process.load("RecoTracker.MeasurementDet.MeasurementTrackerEventProducer_cfi")
     # process.MeasurementTrackerEvent.pixelClusterProducer = 'TRACKTYPETEMPLATE'
     # process.MeasurementTrackerEvent.stripClusterProducer = 'TRACKTYPETEMPLATE'
     # process.MeasurementTrackerEvent.inactivePixelDetectorLabels = cms.VInputTag()
     # process.MeasurementTrackerEvent.inactiveStripDetectorLabels = cms.VInputTag()

     ####################################################################
     # Load and Configure TrackRefitter
     ####################################################################
     process.load("RecoTracker.TrackProducer.TrackRefitters_cff")
     import RecoTracker.TrackProducer.TrackRefitters_cff
     process.FinalTrackRefitter = RecoTracker.TrackProducer.TrackRefitter_cfi.TrackRefitter.clone()
     process.FinalTrackRefitter.src = "TRACKTYPETEMPLATE"
     process.FinalTrackRefitter.TrajectoryInEvent = True
     process.FinalTrackRefitter.NavigationSchool = ''
     process.FinalTrackRefitter.TTRHBuilder = "TTRHBUILDERTEMPLATE"

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
process.TFileService = cms.Service("TFileService",fileName=cms.string("OUTFILETEMPLATE"))

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
# Configure the PVValidation Analyzer module
####################################################################
process.PVValidation = cms.EDAnalyzer("PrimaryVertexValidation",
                                      numberOfBins = cms.untracked.int32(48),
                                      TrackCollectionTag = cms.InputTag("FinalTrackRefitter"),
                                      VertexCollectionTag = cms.InputTag("VERTEXTYPETEMPLATE"),
                                      Debug = cms.bool(False),
                                      storeNtuple = cms.bool(False),
                                      useTracksFromRecoVtx = cms.bool(False),
                                      isLightNtuple = cms.bool(True),
                                      askFirstLayerHit = cms.bool(False),
                                      forceBeamSpot = cms.untracked.bool(False),
                                      probePt = cms.untracked.double(PTCUTTEMPLATE),
                                      probeEta = cms.untracked.double(2.7),
                                      runControl = cms.untracked.bool(RUNCONTROLTEMPLATE),
                                      intLumi = cms.untracked.double(INTLUMITEMPLATE),
                                      runControlNumber = cms.untracked.vuint32(int(runboundary)),
                                      TkFilterParameters = FilteringParams,
                                      TkClusParameters = switchClusterizerParameters(isDA)
                                      )

####################################################################
# Path
####################################################################
process.p = cms.Path(process.goodvertexSkim*
                     # in case the common refitting sequence is removed
                     #process.offlineBeamSpot*
                     process.seqTrackselRefit*
                     # in case the navigation shool is removed
                     #process.MeasurementTrackerEvent*
                     # in case the common refitting sequence is removed
                     #process.TrackRefitter*
                     process.PVValidation)
