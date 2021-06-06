from __future__ import print_function
import FWCore.ParameterSet.Config as cms
import sys
 
isDA = ISDATEMPLATE
isMC = ISMCTEMPLATE
allFromGT = ALLFROMGTTEMPLATE
applyBows = APPLYBOWSTEMPLATE
applyExtraConditions = EXTRACONDTEMPLATE
useFileList = USEFILELISTTEMPLATE

process = cms.Process("PrimaryVertexValidation") 

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
if (useFileList):
     print(">>>>>>>>>> testPVValidation_cfg.py: msg%-i: Reading local input files list")
     readFiles = cms.untracked.vstring()
     readFiles.extend(FILESOURCETEMPLATE)
     process.source = cms.Source("PoolSource",
                                 fileNames = readFiles ,
                                 duplicateCheckMode = cms.untracked.string('checkAllFilesOpened')
                                 )
else:
     print(">>>>>>>>>> testPVValidation_cfg.py: msg%-i: Reading from configuration fragment")
     process.load("Alignment.OfflineValidation.DATASETTEMPLATE")

###################################################################
#  Runs and events
###################################################################
runboundary = RUNBOUNDARYTEMPLATE
process.source.firstRun = cms.untracked.uint32(int(runboundary))
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(MAXEVENTSTEMPLATE) )

###################################################################
# JSON Filtering
###################################################################
if isMC:
     print(">>>>>>>>>> testPVValidation_cfg.py: msg%-i: This is simulation!")
     runboundary = 1
else:
     print(">>>>>>>>>> testPVValidation_cfg.py: msg%-i: This is real DATA!")
     if ('LUMILISTTEMPLATE'):
          print(">>>>>>>>>> testPVValidation_cfg.py: msg%-i: JSON filtering with: LUMILISTTEMPLATE")
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
# Geometry load
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
     print(">>>>>>>>>> testPVValidation_cfg.py: msg%-i: All is taken from GT")
else:
     ####################################################################
     # Get Alignment and APE constants
     ####################################################################
     process=customiseAlignmentAndAPE(process)

     ####################################################################
     # Kinks and Bows (optional)
     ####################################################################
     if applyBows:
          print(">>>>>>>>>> testPVValidation_cfg.py: msg%-i: Applying TrackerSurfaceDeformations!")
          process=customiseKinksAndBows(process)
     else:
          print(">>>>>>>>>> testPVValidation_cfg.py: msg%-i: MultiPVValidation: Not applying TrackerSurfaceDeformations!")

          ####################################################################
          # Extra corrections not included in the GT
          ####################################################################
          if applyExtraConditions:
               print(">>>>>>>>>> testPVValidation_cfg.py: msg%-i: Applying extra calibration constants!")

               import CalibTracker.Configuration.Common.PoolDBESSource_cfi

               # Extra conditions to be plugged here
               ##### END OF EXTRA CONDITIONS
               
          else:
               print(">>>>>>>>>> testPVValidation_cfg.py: msg%-i: Not applying extra calibration constants!")
               
     
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
     process.goodvertexSkim = cms.Sequence(process.noscraping + process.filterOutLowPt)
else:
     process.goodvertexSkim = cms.Sequence(process.primaryVertexFilter + process.noscraping + process.filterOutLowPt)

####################################################################
# Load and Configure Measurement Tracker Event
# (this would be needed in case NavigationSchool is set != from ''
####################################################################
#process.load("RecoTracker.MeasurementDet.MeasurementTrackerEventProducer_cfi") 
#process.MeasurementTrackerEvent.pixelClusterProducer = 'TRACKTYPETEMPLATE'
#process.MeasurementTrackerEvent.stripClusterProducer = 'TRACKTYPETEMPLATE'
#process.MeasurementTrackerEvent.inactivePixelDetectorLabels = cms.VInputTag()
#process.MeasurementTrackerEvent.inactiveStripDetectorLabels = cms.VInputTag()

####################################################################
# Load and Configure TrackRefitter
####################################################################
process.load("RecoTracker.TrackProducer.TrackRefitters_cff")
import RecoTracker.TrackProducer.TrackRefitters_cff
process.TrackRefitter = RecoTracker.TrackProducer.TrackRefitter_cfi.TrackRefitter.clone()
process.TrackRefitter.src = "TRACKTYPETEMPLATE"
process.TrackRefitter.TrajectoryInEvent = True
process.TrackRefitter.NavigationSchool = ''
process.TrackRefitter.TTRHBuilder = "WithAngleAndTemplate"

####################################################################
# Output file
####################################################################
process.TFileService = cms.Service("TFileService",
                                   fileName=cms.string("OUTFILETEMPLATE")
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
          print(">>>>>>>>>> testPVValidation_cfg.py: msg%-i: Running DA Algorithm!")
          return DAClusterizationParams
     else:
          print(">>>>>>>>>> testPVValidation_cfg.py: msg%-i: Running GAP Algorithm!")
          return GapClusterizationParams

####################################################################
# Configure the PVValidation Analyzer module
####################################################################
process.PVValidation = cms.EDAnalyzer("PrimaryVertexValidation",
                                      TrackCollectionTag = cms.InputTag("TrackRefitter"),
                                      VertexCollectionTag = cms.InputTag("VERTEXTYPETEMPLATE"),
                                      Debug = cms.bool(False),
                                      storeNtuple = cms.bool(False),
                                      useTracksFromRecoVtx = cms.bool(False),
                                      isLightNtuple = cms.bool(True),
                                      askFirstLayerHit = cms.bool(False),
                                      forceBeamSpot = cms.untracked.bool(False),
                                      probePt = cms.untracked.double(PTCUTTEMPLATE),
                                      runControl = cms.untracked.bool(RUNCONTROLTEMPLATE),
                                      runControlNumber = cms.untracked.vuint32(int(runboundary)),
                                      TkFilterParameters = FilteringParams,
                                      TkClusParameters = switchClusterizerParameters(isDA)
                                      )

####################################################################
# Path
####################################################################
process.p = cms.Path(process.goodvertexSkim*
                     process.offlineBeamSpot*
                     process.TrackRefitter*
                     process.PVValidation)
