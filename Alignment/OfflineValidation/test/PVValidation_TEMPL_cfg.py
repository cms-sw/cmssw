import FWCore.ParameterSet.Config as cms

isDA = ISDATEMPLATE
applyBows = APPLYBOWSTEMPLATE
applyExtraCorrection = EXTRACORRTEMPLATE

process = cms.Process("Demo")

process.load("Alignment.OfflineValidation.DATASETTEMPLATE");
process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.destinations = ['cout', 'cerr']
process.MessageLogger.cerr.FwkReport.reportEvery = 1000
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(MAXEVENTSTEMPLATE) )

 ##
 ## Get the Magnetic Field
 ##
process.load('Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff')

 ##
 ## Get the Geometry
 ##
from Geometry.CommonDetUnit.globalTrackingGeometry_cfi import *
process.load("Configuration.Geometry.GeometryIdeal_cff")
process.load("Geometry.CommonDetUnit.globalTrackingGeometry_cfi")
process.load("Geometry.TrackerGeometryBuilder.trackerGeometry_cfi")

 ##
 ## Load Global Position Record
 ##
# process.globalPosition = cms.ESSource("PoolDBESSource",CondDBSetup,
#                       toGet = cms.VPSet(cms.PSet(
#                       record = cms.string('GlobalPositionRcd'),
#                       tag= cms.string('IdealGeometry')
#                        )),
#                       connect =cms.string('frontier://FrontierProd/CMS_COND_31X_FROM21X')
#                       )
# process.es_prefer_GPRcd = cms.ESPrefer("PoolDBESSource","globalPosition")


 ##
 ## Load Beamspot
 ##
# process.beamspot = cms.ESSource("PoolDBESSource",CondDBSetup,
#                                 toGet = cms.VPSet(cms.PSet( record = cms.string('BeamSpotObjectsRcd'),
#                                                             tag= cms.string('Realistic7TeVCollisions2011_START311_V2_v2_mc')
#                                                             )),
#                                 connect =cms.string('frontier://FrontierProd/CMS_COND_31X_BEAMSPOT')
#                                 )
# process.es_prefer_beamspot = cms.ESPrefer("PoolDBESSource","beamspot")

 ##
 ## Get the BeamSpot
 ##
process.load("RecoVertex.BeamSpotProducer.BeamSpot_cff")

 ##
 ## Get the GlogalTag
 ##
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = "GLOBALTAGTEMPLATE"  # take your favourite

 ##
 ## Get Alignment constants
 ##
from CondCore.DBCommon.CondDBSetup_cfi import *
process.trackerAlignment = cms.ESSource("PoolDBESSource",CondDBSetup,
                                        connect = cms.string('ALIGNOBJTEMPLATE'),
                                        timetype = cms.string("runnumber"),
                                        toGet = cms.VPSet(cms.PSet(record = cms.string('TrackerAlignmentRcd'),
                                                                   tag = cms.string('GEOMTAGTEMPLATE')
                                                                   ))
                                        )
process.es_prefer_trackerAlignment = cms.ESPrefer("PoolDBESSource", "trackerAlignment")

 ##
 ## Get APE
 ##
process.setAPE = cms.ESSource("PoolDBESSource",CondDBSetup,
                                        connect = cms.string('APEOBJTEMPLATE'),
                                        timetype = cms.string("runnumber"),
                                        toGet = cms.VPSet(cms.PSet(record = cms.string('TrackerAlignmentErrorRcd'),
                                                                   tag = cms.string('ERRORTAGTEMPLATE')
                                                                   ))
                                        )
process.es_prefer_setAPE = cms.ESPrefer("PoolDBESSource", "setAPE")

 ##
 ## Kinks and Bows (optional)
 ##
if applyBows:
     process.trackerBows = cms.ESSource("PoolDBESSource",CondDBSetup,
                                        connect = cms.string('BOWSOBJECTTEMPLATE'),
                                        toGet = cms.VPSet(cms.PSet(record = cms.string('TrackerSurfaceDeformationRcd'),
                                                                   tag = cms.string('BOWSTAGTEMPLATE')
                                                                   )
                                                          )
                                        )
     process.es_prefer_Bows = cms.ESPrefer("PoolDBESSource", "trackerBows")
else:
     print "msg%-i: Primary Vertex Validation: Not applying bows!"

 ##
 ## Extra corrections not included in the GT
 ##
if applyExtraCorrection:
     import CalibTracker.Configuration.Common.PoolDBESSource_cfi

     ## pixel templates
     process.conditionsInSiPixelTemplateDBObjectRcd = CalibTracker.Configuration.Common.PoolDBESSource_cfi.poolDBESSource.clone(
          connect = cms.string('frontier://FrontierProd/CMS_COND_31X_PIXEL'),
          toGet = cms.VPSet(cms.PSet(record = cms.string('SiPixelTemplateDBObjectRcd'),
                                     tag = cms.string('SiPixelTemplateDBObject_38T_v4_offline')
                                     )
                            )
          )
     process.prefer_conditionsInSiPixelTemplateDBObjectRcd = cms.ESPrefer("PoolDBESSource", "conditionsInSiPixelTemplateDBObjectRcd")

     ## SiStrip Lorentz Angle corrections
     process.conditionsInSiStripLorentzAngleRcd = CalibTracker.Configuration.Common.PoolDBESSource_cfi.poolDBESSource.clone(
          connect = cms.string('sqlite_file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERALIGN/MP/MPproduction/mp1025/SiStripLorentzAngleDeco_GR10_v1_offline_BPCorrected.db'),
          toGet = cms.VPSet(cms.PSet(record = cms.string('SiStripLorentzAngleRcd'),
                                     tag = cms.string('SiStripLorentzAngleDeco_GR10_v1_offline_BPCorrected'),
                                     label = cms.untracked.string('deconvolution')
                                     )
                            )
          )
     
     process.prefer_conditionsInSiStripLorentzAngleRcd = cms.ESPrefer("PoolDBESSource", "conditionsInSiStripLorentzAngleRcd")
     
     ## SiStrip backplane corrections
     process.conditionsInSiStripConfObjectRcd = CalibTracker.Configuration.Common.PoolDBESSource_cfi.poolDBESSource.clone(
          connect = cms.string('sqlite_file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERALIGN/MP/MPproduction/mp1025/SiStripShiftAndCrosstalk_GR10_v2_offline_BPCorrected.db'),
          toGet = cms.VPSet(cms.PSet(record = cms.string('SiStripConfObjectRcd'),
                                     tag = cms.string('SiStripShiftAndCrosstalk_GR10_v2_offline_BPCorrected')
                                     )
                            )
          )
     process.prefer_conditionsInSiStripConfObjectRcd = cms.ESPrefer("PoolDBESSource", "conditionsInSiStripConfObjectRcd")
     
else:
     print "msg%-i: Primary Vertex Validation: Not applying extra calibration constants!"
     
 ##
 ## Load and Configure event selection
 ##
process.primaryVertexFilter = cms.EDFilter("VertexSelector",
    src = cms.InputTag("offlinePrimaryVertices"),
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

process.goodvertexSkim = cms.Sequence(process.primaryVertexFilter + process.noscraping)

 ##
 ## output file
 ##
process.TFileService = cms.Service("TFileService",
                                   fileName=cms.string("OUTFILETEMPLATE")
                                   )

 ##
 ## Load and Configure TrackRefitter
 ##
process.load("RecoTracker.TrackProducer.TrackRefitters_cff")
import RecoTracker.TrackProducer.TrackRefitters_cff
process.TrackRefitter = RecoTracker.TrackProducer.TrackRefitter_cfi.TrackRefitter.clone()
process.TrackRefitter.src = "TRACKTYPETEMPLATE"
process.TrackRefitter.TrajectoryInEvent = True
process.TrackRefitter.TTRHBuilder = "WithAngleAndTemplate"
process.TrackRefitter.NavigationSchool = ''

if isDA:  
     process.PVValidation = cms.EDAnalyzer("PrimaryVertexValidation",
                                           TrackCollectionTag = cms.InputTag("TrackRefitter"),
                                           Debug = cms.bool(False),
                                           storeNtuple = cms.bool(True),
                                           isLightNtuple = cms.bool(True),
                                           useTracksFromRecoVtx = cms.bool(False),
                                           askFirstLayerHit = cms.bool(True),
                                           probePt = cms.untracked.double(1.),
                                           probeEta = cms.untracked.double(2.4),
                                           numberOfBins = cms.untracked.int32(24),
                                           
                                           TkFilterParameters = cms.PSet(algorithm=cms.string('filter'),                           
                                                                         maxNormalizedChi2 = cms.double(20.0),                       # chi2ndof < 20                  
                                                                         minPixelLayersWithHits=cms.int32(2),                        # PX hits > 2                       
                                                                         minSiliconLayersWithHits = cms.int32(5),                    # TK hits > 5  
                                                                         maxD0Significance = cms.double(1000),                       # fake cut (requiring 1 PXB hit)     
                                                                         minPt = cms.double(1.0),                                    # better for softish events
                                                                         trackQuality = cms.string("any")
                                                                         ),
                                           
                                           TkClusParameters=cms.PSet(algorithm=cms.string('DA'),
                                                                     TkDAClusParameters = cms.PSet(coolingFactor = cms.double(0.6),  # moderate annealing speed
                                                                                                   Tmin = cms.double(4.),            # end of annealing
                                                                                                   vertexSize = cms.double(0.01),    # ~ resolution / sqrt(Tmin)
                                                                                                   d0CutOff = cms.double(10.),       # downweight high IP tracks
                                                                                                   dzCutOff = cms.double(10.)        # outlier rejection after freeze-out (T<Tmin)
                                                                                                   )
                                                                     )
                                           )
     
else:
     process.PVValidation = cms.EDAnalyzer("PrimaryVertexValidation",
                                           TrackCollectionTag = cms.InputTag("TrackRefitter"),
                                           Debug = cms.bool(False),
                                           storeNtuple = cms.bool(True),
                                           isLightNtuple = cms.bool(True),
                                           useTracksFromRecoVtx = cms.bool(False),
                                           askFirstLayerHit = cms.bool(True),
                                           probePt = cms.untracked.double(1.),
                                           probeEta = cms.untracked.double(2.4),
                                           numberOfBins = cms.untracked.int32(24),

                                           TkFilterParameters = cms.PSet(algorithm=cms.string('filter'),                             
                                                                         maxNormalizedChi2 = cms.double(20.0),                       # chi2ndof < 20                  
                                                                         minPixelLayersWithHits=cms.int32(2),                        # PX hits > 2                   
                                                                         minSiliconLayersWithHits = cms.int32(5),                    # TK hits > 5                   
                                                                         maxD0Significance = cms.double(10000.0),                    # fake cut (requiring 1 PXB hit)
                                                                         minPt = cms.double(1.0),                                    # better for softish events     
                                                                         trackQuality = cms.string("any")
                                                                         ),
                                        
                                           TkClusParameters = cms.PSet(algorithm   = cms.string('gap'),
                                                                       TkGapClusParameters = cms.PSet(zSeparation = cms.double(0.2)  # 0.2 cm max separation betw. clusters
                                                                                                      ) 
                                                                       )
                                           )

process.p = cms.Path(process.goodvertexSkim*process.offlineBeamSpot*process.TrackRefitter*process.PVValidation)
