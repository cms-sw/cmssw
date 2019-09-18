from __future__ import print_function
import FWCore.ParameterSet.Config as cms
import sys
 
isDA = ISDATEMPLATE
isMC = ISMCTEMPLATE
allFromGT = ALLFROMGTTEMPLATE
applyBows = APPLYBOWSTEMPLATE
applyExtraConditions = EXTRACONDTEMPLATE

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
readFiles = cms.untracked.vstring()
readFiles.extend(FILESOURCETEMPLATE)
process.source = cms.Source("PoolSource",
                            fileNames = readFiles ,
                            duplicateCheckMode = cms.untracked.string('checkAllFilesOpened')
                            )

#process.load("Alignment.OfflineValidation.DATASETTEMPLATE");
process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.destinations = ['cout', 'cerr']
process.MessageLogger.cerr.FwkReport.reportEvery = 1000

runboundary = RUNBOUNDARYTEMPLATE
process.source.firstRun = cms.untracked.uint32(int(runboundary))
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(MAXEVENTSTEMPLATE) )

###################################################################
# JSON Filtering
###################################################################
if isMC:
     print(">>>>>>>>>> testPVValidation_cfg.py: msg%-i: This is Simulation!")
     runboundary = 1
else:
     print(">>>>>>>>>> testPVValidation_cfg.py: msg%-i: This is DATA!")
     import FWCore.PythonUtilities.LumiList as LumiList
     process.source.lumisToProcess = LumiList.LumiList(filename ='LUMILISTTEMPLATE').getVLuminosityBlockRange()

###################################################################
# Messages
###################################################################
process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.destinations = ['cout', 'cerr']
process.MessageLogger.cerr.FwkReport.reportEvery = 1000

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
process.GlobalTag = GlobalTag(process.GlobalTag, 'GLOBALTAGTEMPLATE', '')

if allFromGT:
     print(">>>>>>>>>> testPVValidation_cfg.py: msg%-i: All is taken from GT")
else:
     ####################################################################
     # Get Alignment constants and APE
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

          import CalibTracker.Configuration.Common.PoolDBESSource_cfi
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
import Alignment.CommonAlignment.tools.trackselectionRefitting as trackselRefit
process.seqTrackselRefit = trackselRefit.getSequence(process,'TRACKTYPETEMPLATE')
process.HighPurityTrackSelector.trackQualities = cms.vstring()
process.HighPurityTrackSelector.pMin     = cms.double(0.)
#process.TrackerTrackHitFilter.usePixelQualityFlag = cms.bool(False)    # do not use the pixel quality flag
#process.TrackerTrackHitFilter.commands   = cms.vstring("drop PXB 1")   # drop BPix1 hits
process.AlignmentTrackSelector.pMin      = cms.double(0.)
process.AlignmentTrackSelector.ptMin     = cms.double(0.)
process.AlignmentTrackSelector.nHitMin2D = cms.uint32(0)
process.AlignmentTrackSelector.nHitMin   = cms.double(0.)
process.AlignmentTrackSelector.d0Min     = cms.double(-999999.0)
process.AlignmentTrackSelector.d0Max     = cms.double(+999999.0)                  
process.AlignmentTrackSelector.dzMin     = cms.double(-999999.0)
process.AlignmentTrackSelector.dzMax     = cms.double(+999999.0)  

####################################################################
# Output file
####################################################################
process.TFileService = cms.Service("TFileService",
                                   fileName=cms.string("OUTFILETEMPLATE")
                                  )                                    

####################################################################
# Deterministic annealing clustering
####################################################################
if isDA:
     print(">>>>>>>>>> testPVValidation_cfg.py: msg%-i: Running DA Algorithm!")
     process.PVValidation = cms.EDAnalyzer("PrimaryVertexValidation",
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
     print(">>>>>>>>>> testPVValidation_cfg.py: msg%-i: Running GAP Algorithm!")
     process.PVValidation = cms.EDAnalyzer("PrimaryVertexValidation",
                                           TrackCollectionTag = cms.InputTag("FinalTrackRefitter"),
                                           VertexCollectionTag = cms.InputTag("VERTEXTYPETEMPLATE"), 
                                           Debug = cms.bool(False),
                                           isLightNtuple = cms.bool(True),
                                           storeNtuple = cms.bool(False),
                                           useTracksFromRecoVtx = cms.bool(False),
                                           askFirstLayerHit = cms.bool(False),
                                           forceBeamSpot = cms.untracked.bool(False),
                                           probePt = cms.untracked.double(PTCUTTEMPLATE),
                                           probeEta = cms.untracked.double(2.7),
                                           runControl = cms.untracked.bool(RUNCONTROLTEMPLATE),
                                           intLumi = cms.untracked.double(INTLUMITEMPLATE),
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
                     # in case the common refitting sequence is removed
                     #process.offlineBeamSpot*
                     process.seqTrackselRefit*
                     # in case the navigation shool is removed
                     #process.MeasurementTrackerEvent*
                     # in case the common refitting sequence is removed
                     #process.TrackRefitter*
                     process.PVValidation)
