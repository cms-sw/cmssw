import os
import FWCore.ParameterSet.Config as cms

inputfiles = os.environ["ALIGNMENT_INPUTFILES"].split(" ")
iteration = int(os.environ["ALIGNMENT_ITERATION"])
jobnumber = int(os.environ["ALIGNMENT_JOBNUMBER"])
mapplots = (os.environ["ALIGNMENT_MAPPLOTS"] == "True")
segdiffplots = (os.environ["ALIGNMENT_SEGDIFFPLOTS"] == "True")
curvatureplots = (os.environ["ALIGNMENT_CURVATUREPLOTS"] == "True")

globaltag = os.environ["ALIGNMENT_GLOBALTAG"]
inputdb = os.environ["ALIGNMENT_INPUTDB"]
trackerconnect = os.environ["ALIGNMENT_TRACKERCONNECT"]
trackeralignment = os.environ["ALIGNMENT_TRACKERALIGNMENT"]
trackerAPEconnect = os.environ["ALIGNMENT_TRACKERAPECONNECT"]
trackerAPE = os.environ["ALIGNMENT_TRACKERAPE"]
trackerBowsconnect = os.environ["ALIGNMENT_TRACKERBOWSCONNECT"]
trackerBows = os.environ["ALIGNMENT_TRACKERBOWS"]
gprcdconnect = os.environ["ALIGNMENT_GPRCDCONNECT"]
gprcd = os.environ["ALIGNMENT_GPRCD"]

iscosmics = (os.environ["ALIGNMENT_ISCOSMICS"] == "True")
station123params = os.environ["ALIGNMENT_STATION123PARAMS"]
station4params = os.environ["ALIGNMENT_STATION4PARAMS"]
cscparams = os.environ["ALIGNMENT_CSCPARAMS"]
minTrackPt = float(os.environ["ALIGNMENT_MINTRACKPT"])
maxTrackPt = float(os.environ["ALIGNMENT_MAXTRACKPT"])
minTrackP = float(os.environ["ALIGNMENT_MINTRACKP"])
maxTrackP = float(os.environ["ALIGNMENT_MAXTRACKP"])
minTrackerHits = int(os.environ["ALIGNMENT_MINTRACKERHITS"])
maxTrackerRedChi2 = float(os.environ["ALIGNMENT_MAXTRACKERREDCHI2"])
allowTIDTEC = (os.environ["ALIGNMENT_ALLOWTIDTEC"] == "True")
twoBin = (os.environ["ALIGNMENT_TWOBIN"] == "True")
weightAlignment = (os.environ["ALIGNMENT_WEIGHTALIGNMENT"] == "True")
minAlignmentHits = int(os.environ["ALIGNMENT_MINALIGNMENTHITS"])
combineME11 = (os.environ["ALIGNMENT_COMBINEME11"] == "True")
maxEvents = int(os.environ["ALIGNMENT_MAXEVENTS"])
skipEvents = int(os.environ["ALIGNMENT_SKIPEVENTS"])
maxResSlopeY = float(os.environ["ALIGNMENT_MAXRESSLOPEY"])
preFilter = (os.environ["ALIGNMENT_PREFILTER"] == "True")
muonCollectionTag = os.environ["ALIGNMENT_MUONCOLLECTIONTAG"]
maxDxy = float(os.environ["ALIGNMENT_MAXDXY"])
minNCrossedChambers = int(os.environ["ALIGNMENT_MINNCROSSEDCHAMBERS"])
T0_Corr = (os.environ["ALIGNMENT_T0CORR"] == "True")
is_Alcareco = (os.environ["ALIGNMENT_ISALCARECO"] == "True")
is_MC = (os.environ["ALIGNMENT_ISMC"] == "True")
createLayerNtupleDT = (os.environ["ALIGNMENT_STORELAYERDT"] == "True")
createLayerNtupleCSC = (os.environ["ALIGNMENT_STORELAYERCSC"] == "True")

# optionally: create ntuples along with tmp files
createAlignNtuple = False
envNtuple = os.getenv("ALIGNMENT_CREATEALIGNNTUPLE")
if envNtuple is not None:
  if envNtuple=='True': createAlignNtuple = True

# optionally: create a ntuple with MapPlot plugin
createMapNtuple = False
envNtuple = os.getenv("ALIGNMENT_CREATEMAPNTUPLE")
if envNtuple is not None:
  if envNtuple=='True': createMapNtuple = True

# optionally do selective DT or CSC alignment
doDT = True
doCSC = True
envDT = os.getenv("ALIGNMENT_DO_DT")
envCSC = os.getenv("ALIGNMENT_DO_CSC")
if envDT is not None and envCSC is not None:
  if envDT=='True' and envCSC=='False':
    doDT = True
    doCSC = False
  if envDT=='False' and envCSC=='True':
    doDT = False
    doCSC = True

from Configuration.Eras.Era_Run3_cff import Run3
process = cms.Process("GATHER", Run3)

process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")
process.load("Geometry.DTGeometry.dtGeometry_cfi")
process.load("Geometry.RPCGeometry.rpcGeometry_cfi")
process.load("Geometry.CSCGeometry.cscGeometry_cfi")
process.load("Geometry.CommonTopologies.bareGlobalTrackingGeometry_cfi")

#add TrackDetectorAssociator lookup maps to the EventSetup
process.load("TrackingTools.TrackAssociator.DetIdAssociatorESProducer_cff")
from TrackingTools.TrackAssociator.DetIdAssociatorESProducer_cff import *
from TrackingTools.TrackAssociator.default_cfi import *


process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load('TrackingTools.TransientTrack.TransientTrackBuilder_cfi')

if is_MC:
    process.load('Configuration.StandardSequences.SimIdeal_cff')
    process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
else:
    process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff')
process.source = cms.Source("PoolSource",
  fileNames = cms.untracked.vstring(*inputfiles),
  skipEvents = cms.untracked.uint32(skipEvents))
json_file = os.getenv("ALIGNMENT_JSON")
if len(json_file) > 0:
  import FWCore.PythonUtilities.LumiList as LumiList
  process.source.lumisToProcess = LumiList.LumiList(filename = json_file).getVLuminosityBlockRange()

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(maxEvents))
#process.options = cms.untracked.PSet(  wantSummary = cms.untracked.bool(True) )


# process.MessageLogger = cms.Service("MessageLogger",
#                                     destinations = cms.untracked.vstring("cout"),
#                                     cout = cms.untracked.PSet(threshold = cms.untracked.string("ERROR"),
#                                                               ERROR = cms.untracked.PSet(limit = cms.untracked.int32(10))))

process.load("Alignment.MuonAlignmentAlgorithms.MuonAlignmentFromReference_cff")
process.looper.ParameterBuilder.Selector.alignParams = cms.vstring("MuonDTChambers,%s,stations123" % station123params, "MuonDTChambers,%s,station4" % station4params, "MuonCSCChambers,%s" % cscparams)
# TODO : uncomment the line below when AlignmentProducer is updated:
#process.looper.muonCollectionTag = cms.InputTag(muonCollectionTag)
process.looper.algoConfig.writeTemporaryFile = "alignment%04d.tmp" % jobnumber
process.looper.algoConfig.doAlignment = False
process.looper.algoConfig.muonCollectionTag = cms.InputTag(muonCollectionTag)
process.looper.algoConfig.minTrackPt = minTrackPt
process.looper.algoConfig.maxTrackPt = maxTrackPt
process.looper.algoConfig.minTrackP = minTrackP
process.looper.algoConfig.maxTrackP = maxTrackP
process.looper.algoConfig.maxDxy = maxDxy
process.looper.algoConfig.minTrackerHits = minTrackerHits
process.looper.algoConfig.maxTrackerRedChi2 = maxTrackerRedChi2
process.looper.algoConfig.allowTIDTEC = allowTIDTEC
process.looper.algoConfig.minNCrossedChambers = minNCrossedChambers
process.looper.algoConfig.twoBin = twoBin
process.looper.algoConfig.weightAlignment = weightAlignment
process.looper.algoConfig.minAlignmentHits = minAlignmentHits
process.looper.algoConfig.combineME11 = combineME11
process.looper.algoConfig.maxResSlopeY = maxResSlopeY
#process.looper.algoConfig.createNtuple = createAlignNtuple
process.looper.algoConfig.minDT13Hits = 7
process.looper.algoConfig.doDT = doDT
process.looper.algoConfig.doCSC = doCSC
process.looper.algoConfig.createLayerNtupleDT = createLayerNtupleDT
process.looper.algoConfig.createLayerNtupleCSC = createLayerNtupleCSC

process.looper.monitorConfig = cms.PSet(monitors = cms.untracked.vstring())

if mapplots:
    process.load("Alignment.CommonAlignmentMonitor.AlignmentMonitorMuonSystemMap1D_cfi")
    process.looper.monitorConfig.monitors.append("AlignmentMonitorMuonSystemMap1D")
    process.looper.monitorConfig.AlignmentMonitorMuonSystemMap1D = process.AlignmentMonitorMuonSystemMap1D
    process.looper.monitorConfig.AlignmentMonitorMuonSystemMap1D.muonCollectionTag = cms.InputTag(muonCollectionTag)
    process.looper.monitorConfig.AlignmentMonitorMuonSystemMap1D.minTrackPt = minTrackPt
    process.looper.monitorConfig.AlignmentMonitorMuonSystemMap1D.maxTrackPt = maxTrackPt
    process.looper.monitorConfig.AlignmentMonitorMuonSystemMap1D.minTrackP = minTrackP
    process.looper.monitorConfig.AlignmentMonitorMuonSystemMap1D.maxTrackP = maxTrackP
    process.looper.monitorConfig.AlignmentMonitorMuonSystemMap1D.maxDxy = maxDxy
    process.looper.monitorConfig.AlignmentMonitorMuonSystemMap1D.minTrackerHits = minTrackerHits
    process.looper.monitorConfig.AlignmentMonitorMuonSystemMap1D.maxTrackerRedChi2 = maxTrackerRedChi2
    process.looper.monitorConfig.AlignmentMonitorMuonSystemMap1D.allowTIDTEC = allowTIDTEC
    process.looper.monitorConfig.AlignmentMonitorMuonSystemMap1D.minNCrossedChambers = process.looper.algoConfig.minNCrossedChambers
    process.looper.monitorConfig.AlignmentMonitorMuonSystemMap1D.minDT13Hits = process.looper.algoConfig.minDT13Hits
    process.looper.monitorConfig.AlignmentMonitorMuonSystemMap1D.minDT2Hits = process.looper.algoConfig.minDT2Hits
    process.looper.monitorConfig.AlignmentMonitorMuonSystemMap1D.minCSCHits = process.looper.algoConfig.minCSCHits
    process.looper.monitorConfig.AlignmentMonitorMuonSystemMap1D.doDT = doDT
    process.looper.monitorConfig.AlignmentMonitorMuonSystemMap1D.doCSC = doCSC
    process.looper.monitorConfig.AlignmentMonitorMuonSystemMap1D.createNtuple = createMapNtuple

if segdiffplots:
    process.load("Alignment.CommonAlignmentMonitor.AlignmentMonitorSegmentDifferences_cfi")
    process.looper.monitorConfig.monitors.append("AlignmentMonitorSegmentDifferences")
    process.looper.monitorConfig.AlignmentMonitorSegmentDifferences = process.AlignmentMonitorSegmentDifferences
    process.looper.monitorConfig.AlignmentMonitorSegmentDifferences.muonCollectionTag = cms.InputTag(muonCollectionTag)
    process.looper.monitorConfig.AlignmentMonitorSegmentDifferences.minTrackPt = minTrackPt
    process.looper.monitorConfig.AlignmentMonitorSegmentDifferences.minTrackP = minTrackP
    process.looper.monitorConfig.AlignmentMonitorSegmentDifferences.maxDxy = maxDxy
    process.looper.monitorConfig.AlignmentMonitorSegmentDifferences.minTrackerHits = minTrackerHits
    process.looper.monitorConfig.AlignmentMonitorSegmentDifferences.maxTrackerRedChi2 = maxTrackerRedChi2
    process.looper.monitorConfig.AlignmentMonitorSegmentDifferences.allowTIDTEC = allowTIDTEC
    process.looper.monitorConfig.AlignmentMonitorSegmentDifferences.minNCrossedChambers = process.looper.algoConfig.minNCrossedChambers
    process.looper.monitorConfig.AlignmentMonitorSegmentDifferences.minDT13Hits = process.looper.algoConfig.minDT13Hits
    process.looper.monitorConfig.AlignmentMonitorSegmentDifferences.minDT2Hits = process.looper.algoConfig.minDT2Hits
    process.looper.monitorConfig.AlignmentMonitorSegmentDifferences.minCSCHits = process.looper.algoConfig.minCSCHits
    process.looper.monitorConfig.AlignmentMonitorSegmentDifferences.doDT = doDT
    process.looper.monitorConfig.AlignmentMonitorSegmentDifferences.doCSC = doCSC

if curvatureplots:
    process.load("Alignment.CommonAlignmentMonitor.AlignmentMonitorMuonVsCurvature_cfi")
    process.looper.monitorConfig.monitors.append("AlignmentMonitorMuonVsCurvature")
    process.looper.monitorConfig.AlignmentMonitorMuonVsCurvature = process.AlignmentMonitorMuonVsCurvature
    process.looper.monitorConfig.AlignmentMonitorMuonVsCurvature.muonCollectionTag = cms.InputTag(muonCollectionTag)
    process.looper.monitorConfig.AlignmentMonitorMuonVsCurvature.minTrackPt = minTrackPt
    #process.looper.monitorConfig.AlignmentMonitorMuonVsCurvature.minTrackP = minTrackP
    process.looper.monitorConfig.AlignmentMonitorMuonVsCurvature.maxDxy = maxDxy
    process.looper.monitorConfig.AlignmentMonitorMuonVsCurvature.minTrackerHits = minTrackerHits
    process.looper.monitorConfig.AlignmentMonitorMuonVsCurvature.maxTrackerRedChi2 = maxTrackerRedChi2
    process.looper.monitorConfig.AlignmentMonitorMuonVsCurvature.allowTIDTEC = allowTIDTEC
    process.looper.monitorConfig.AlignmentMonitorMuonVsCurvature.minNCrossedChambers = process.looper.algoConfig.minNCrossedChambers
    process.looper.monitorConfig.AlignmentMonitorMuonVsCurvature.minDT13Hits = process.looper.algoConfig.minDT13Hits
    process.looper.monitorConfig.AlignmentMonitorMuonVsCurvature.minDT2Hits = process.looper.algoConfig.minDT2Hits
    process.looper.monitorConfig.AlignmentMonitorMuonVsCurvature.minCSCHits = process.looper.algoConfig.minCSCHits
    process.looper.monitorConfig.AlignmentMonitorMuonVsCurvature.doDT = doDT
    process.looper.monitorConfig.AlignmentMonitorMuonVsCurvature.doCSC = doCSC

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
print(f'Using global tag: {globaltag}')
process.GlobalTag.globaltag = cms.string(globaltag)
process.looper.applyDbAlignment = True
process.load("RecoVertex.BeamSpotProducer.BeamSpot_cfi")

process.load("Alignment.MuonAlignmentAlgorithms.MuonAlignmentPreFilter_cfi")
process.MuonAlignmentPreFilter.minTrackPt = minTrackPt
process.MuonAlignmentPreFilter.minTrackP = minTrackP
process.MuonAlignmentPreFilter.minTrackerHits = minTrackerHits
process.MuonAlignmentPreFilter.allowTIDTEC = allowTIDTEC

##T0 Correction on DT need GlobalMuons to be reconstructed
if T0_Corr:
    process.load("RecoLocalMuon.Configuration.RecoLocalMuon_cff")
    process.load("RecoMuon.MuonSeedGenerator.ancientMuonSeed_cfi")
    process.load("RecoMuon.StandAloneMuonProducer.standAloneMuons_cfi")
    process.load("RecoMuon.GlobalMuonProducer.globalMuons_cfi")
    if is_Alcareco:
        process.globalMuons.TrackerCollectionLabel = cms.InputTag("ALCARECOMuAlCalIsolatedMuGeneralTracks")
    else:
        process.globalMuons.TrackerCollectionLabel = cms.InputTag("generalTracks")
    process.Mymuonlocalreco = cms.Sequence(process.dt4DSegments * process.ancientMuonSeed * process.standAloneMuons * process.globalMuons )
    process.dt4DSegments.Reco4DAlgoConfig.performT0SegCorrection = cms.bool(True)


if iscosmics:
    process.MuonAlignmentPreFilter.tracksTag = cms.InputTag("ALCARECOMuAlGlobalCosmics:GlobalMuon")
    if preFilter: process.Path = cms.Path(process.offlineBeamSpot * process.MuonAlignmentPreFilter * process.MuonAlignmentFromReferenceGlobalCosmicRefit)
    else: process.Path = cms.Path(process.offlineBeamSpot * process.MuonAlignmentFromReferenceGlobalCosmicRefit)
    process.looper.tjTkAssociationMapTag = cms.InputTag("MuonAlignmentFromReferenceGlobalCosmicRefit:Refitted")
else:
    if is_Alcareco:
      process.MuonAlignmentPreFilter.tracksTag = cms.InputTag("ALCARECOMuAlCalIsolatedMu:GlobalMuon")
      process.MuonAlignmentFromReferenceGlobalMuonRefit.Tracks = cms.InputTag("ALCARECOMuAlCalIsolatedMu:GlobalMuon")
    else:
      process.MuonAlignmentPreFilter.tracksTag = cms.InputTag("globalMuons")
      process.MuonAlignmentFromReferenceGlobalMuonRefit.Tracks = cms.InputTag("globalMuons")
    process.MuonAlignmentFromReferenceGlobalMuonRefit.Tracks = cms.InputTag("globalMuons")
    if preFilter:
      process.Path = cms.Path(process.offlineBeamSpot * process.MuonAlignmentPreFilter * process.MuonAlignmentFromReferenceGlobalMuonRefit)
    else:
      if T0_Corr:
        process.Path = cms.Path(process.offlineBeamSpot * process.Mymuonlocalreco * process.MuonAlignmentFromReferenceGlobalMuonRefit)
      else:
        process.Path = cms.Path(process.offlineBeamSpot * process.MuonAlignmentFromReferenceGlobalMuonRefit)
    process.looper.tjTkAssociationMapTag = cms.InputTag("MuonAlignmentFromReferenceGlobalMuonRefit:Refitted")


if len(muonCollectionTag) > 0: # use Tracker Muons
    process.Path = cms.Path(process.offlineBeamSpot * process.newmuons)


process.MuonAlignmentFromReferenceInputDB.connect = cms.string("sqlite_file:%s" % inputdb)
process.MuonAlignmentFromReferenceInputDB.toGet = cms.VPSet(cms.PSet(record = cms.string("DTAlignmentRcd"), tag = cms.string("DTAlignmentRcd")),
                                                            cms.PSet(record = cms.string("CSCAlignmentRcd"), tag = cms.string("CSCAlignmentRcd")))

process.es_prefer_MuonAlignmentFromReferenceInputDB = cms.ESPrefer(
    "PoolDBESSource", "MuonAlignmentFromReferenceInputDB")

from CondCore.CondDB.CondDB_cfi import *
CondDBSetup = CondDB.clone()
CondDBSetup.__delattr__('connect')
if is_MC:
    if trackerconnect != "":
        process.TrackerAlignmentInputDB = cms.ESSource("PoolDBESSource",
                                                       CondDBSetup,
                                                       connect = cms.string(trackerconnect),
                                                       toGet = cms.VPSet(cms.PSet(record = cms.string("TrackerAlignmentRcd"), tag = cms.string(trackeralignment))))
        process.es_prefer_TrackerAlignmentInputDB = cms.ESPrefer("PoolDBESSource", "TrackerAlignmentInputDB")

    if trackerAPEconnect != "":
        process.TrackerAlignmentErrorInputDB = cms.ESSource("PoolDBESSource",
                                                       CondDBSetup,
                                                       connect = cms.string(trackerAPEconnect),
                                                       toGet = cms.VPSet(cms.PSet(cms.PSet(record = cms.string("TrackerAlignmentErrorExtendedRcd"), tag = cms.string(trackerAPE)))))
        process.es_prefer_TrackerAlignmentErrorInputDB = cms.ESPrefer("PoolDBESSource", "TrackerAlignmentErrorInputDB")

    if trackerBowsconnect != "":
        process.TrackerSurfaceDeformationInputDB = cms.ESSource("PoolDBESSource",
                                                       CondDBSetup,
                                                       connect = cms.string(trackerBowsconnect),
                                                       toGet = cms.VPSet(cms.PSet(cms.PSet(record = cms.string("TrackerSurfaceDeformationRcd"), tag = cms.string(trackerBows)))))
        process.es_prefer_TrackerSurfaceDeformationInputDB = cms.ESPrefer("PoolDBESSource", "TrackerSurfaceDeformationInputDB")
#else: #beginning 2016-rereco  (ALL in 80X_dataRun2_2016LegacyRepro_Candidate_v0)
#    process.GlobalTag.toGet = cms.VPSet(
#             cms.PSet(record = cms.string("TrackerAlignmentRcd"),
#                      tag =  cms.string("TrackerAlignment_EOY16_sm1959"),
#                      connect = cms.string('frontier://FrontierProd/CMS_CONDITIONS')
#                      ),
###             cms.PSet(record = cms.string("TrackerAlignmentErrorExtendedRcd"),
###                      tag =  cms.string("TrackerAlignmentExtendedErrors_MP_Run2016B"),
###                      connect = cms.string('frontier://FrontierProd/CMS_CONDITIONS')
###                      ),
#             cms.PSet(record = cms.string("SiPixelTemplateDBObjectRcd"),
#                      tag =  cms.string("SiPixelTemplateDBObject_38T_v10_offline"),
#                      connect = cms.string('frontier://FrontierProd/CMS_CONDITIONS')
#                      ),
#             cms.PSet(record = cms.string("TrackerSurfaceDeformationRcd"),
#                      tag =  cms.string("TrackerSurfaceDeformations_EOY16_mp2269"),
#                      connect = cms.string('frontier://FrontierProd/CMS_CONDITIONS')
#                      )
#    )
if gprcdconnect != "":
   process.GlobalPositionInputDB = cms.ESSource("PoolDBESSource",
                                                  CondDBSetup,
                                                  connect = cms.string(gprcdconnect),
                                                  toGet = cms.VPSet(cms.PSet(record = cms.string("GlobalPositionRcd"), tag = cms.string(gprcd))))
   process.es_prefer_GlobalPositionInputDB = cms.ESPrefer("PoolDBESSource", "GlobalPositionInputDB")


process.looper.saveToDB = False
process.looper.saveApeToDB = False
del process.PoolDBOutputService

process.TFileService = cms.Service("TFileService", fileName = cms.string("plotting%03d.root" % jobnumber))

maxEvts = process.maxEvents.input.value()
if maxEvts > 10000 or maxEvts < 0:
  process.MessageLogger.cerr.FwkReport.reportEvery = 1000
elif maxEvts > 10:
  process.MessageLogger.cerr.FwkReport.reportEvery = maxEvts//10

# process.MessageLogger.cerr.threshold = "DEBUG"
# process.MessageLogger.cerr.INFO = cms.untracked.PSet(
#     limit = cms.untracked.int32(-1)   # -1 = unlimited
# )
# process.MessageLogger.cerr.default = cms.untracked.PSet(
#     limit = cms.untracked.int32(-1)   # -1 = unlimited
# )

# process.Tracer = cms.Service("Tracer")

# print(process.dumpPython())
