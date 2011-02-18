import os
import FWCore.ParameterSet.Config as cms

# for json support
try: # FUTURE: Python 2.6, prior to 2.6 requires simplejson
    import json
except:
    try:
        import simplejson as json
    except:
        print "Please use lxplus or set an environment (for example crab) with json lib available"
        sys.exit(1)

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

# optionally use JSON file for good limi mask
good_lumis = []
json_file = os.getenv("ALIGNMENT_JSON")
#json_file = 'Cert_136035-144114_7TeV_StreamExpress_Collisions10_JSON.txt'
if json_file is not None and json_file != '':
  jsonfile=file(json_file, 'r')
  jsondict = json.load(jsonfile)
  runs = jsondict.keys()
  runs.sort()
  for run in runs:
    blocks = jsondict[run]
    blocks.sort()
    prevblock = [-2,-2]
    for lsrange in blocks:
      if lsrange[0] == prevblock[1]+1:
        #print "Run: ",run,"- This lumi starts at ", lsrange[0], " previous ended at ", prevblock[1]+1, " so I should merge"
        prevblock[1] = lsrange[1]
        good_lumis[-1] = str("%s:%s-%s:%s" % (run, prevblock[0], run, prevblock[1]))
      else:
        good_lumis.append(str("%s:%s-%s:%s" % (run, lsrange[0], run, lsrange[1])))
        prevblock = lsrange


process = cms.Process("GATHER")

if len(good_lumis)>0:
  process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(*inputfiles),
    skipEvents = cms.untracked.uint32(skipEvents), 
    lumisToProcess = cms.untracked.VLuminosityBlockRange(*good_lumis))
else:
  process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(*inputfiles),
    skipEvents = cms.untracked.uint32(skipEvents))

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(maxEvents))


process.MessageLogger = cms.Service("MessageLogger",
                                    destinations = cms.untracked.vstring("cout"),
                                    cout = cms.untracked.PSet(threshold = cms.untracked.string("ERROR")))

process.load("Alignment.MuonAlignmentAlgorithms.MuonAlignmentFromReference_cff")
process.looper.algoConfig.writeTemporaryFile = "alignment%04d.tmp" % jobnumber
process.looper.algoConfig.doAlignment = False

process.looper.ParameterBuilder.Selector.alignParams = cms.vstring("MuonDTChambers,%s,stations123" % station123params, "MuonDTChambers,%s,station4" % station4params, "MuonCSCChambers,%s" % cscparams)
process.looper.algoConfig.minTrackPt = minTrackPt
process.looper.algoConfig.maxTrackPt = maxTrackPt
process.looper.algoConfig.minTrackP = minTrackP
process.looper.algoConfig.maxTrackP = maxTrackP
process.looper.algoConfig.minTrackerHits = minTrackerHits
process.looper.algoConfig.maxTrackerRedChi2 = maxTrackerRedChi2
process.looper.algoConfig.allowTIDTEC = allowTIDTEC
process.looper.algoConfig.twoBin = twoBin
process.looper.algoConfig.weightAlignment = weightAlignment
process.looper.algoConfig.minAlignmentHits = minAlignmentHits
process.looper.algoConfig.combineME11 = combineME11
process.looper.algoConfig.maxResSlopeY = maxResSlopeY

process.looper.monitorConfig = cms.PSet(monitors = cms.untracked.vstring())

if mapplots:
    process.load("Alignment.CommonAlignmentMonitor.AlignmentMonitorMuonSystemMap1D_cfi")
    process.looper.monitorConfig.monitors.append("AlignmentMonitorMuonSystemMap1D")
    process.looper.monitorConfig.AlignmentMonitorMuonSystemMap1D = process.AlignmentMonitorMuonSystemMap1D
    process.looper.monitorConfig.AlignmentMonitorMuonSystemMap1D.minTrackPt = cms.double(minTrackPt)
    process.looper.monitorConfig.AlignmentMonitorMuonSystemMap1D.maxTrackPt = cms.double(maxTrackPt)
    process.looper.monitorConfig.AlignmentMonitorMuonSystemMap1D.minTrackerHits = cms.int32(minTrackerHits)
    process.looper.monitorConfig.AlignmentMonitorMuonSystemMap1D.maxTrackerRedChi2 = cms.double(maxTrackerRedChi2)
    process.looper.monitorConfig.AlignmentMonitorMuonSystemMap1D.allowTIDTEC = cms.bool(allowTIDTEC)
    process.looper.monitorConfig.AlignmentMonitorMuonSystemMap1D.minDT13Hits = process.looper.algoConfig.minDT13Hits
    process.looper.monitorConfig.AlignmentMonitorMuonSystemMap1D.minDT2Hits = process.looper.algoConfig.minDT2Hits
    process.looper.monitorConfig.AlignmentMonitorMuonSystemMap1D.minCSCHits = process.looper.algoConfig.minCSCHits
    process.looper.monitorConfig.AlignmentMonitorMuonSystemMap1D.doDT = cms.bool(doDT)
    process.looper.monitorConfig.AlignmentMonitorMuonSystemMap1D.doCSC = cms.bool(doCSC)
    process.looper.monitorConfig.AlignmentMonitorMuonSystemMap1D.createNtuple = cms.bool(createMapNtuple)

if segdiffplots:
    process.load("Alignment.CommonAlignmentMonitor.AlignmentMonitorSegmentDifferences_cfi")
    process.looper.monitorConfig.monitors.append("AlignmentMonitorSegmentDifferences")
    process.looper.monitorConfig.AlignmentMonitorSegmentDifferences = process.AlignmentMonitorSegmentDifferences
    process.looper.monitorConfig.AlignmentMonitorSegmentDifferences.minTrackPt = cms.double(minTrackPt)
    process.looper.monitorConfig.AlignmentMonitorSegmentDifferences.minTrackerHits = cms.int32(minTrackerHits)
    process.looper.monitorConfig.AlignmentMonitorSegmentDifferences.maxTrackerRedChi2 = cms.double(maxTrackerRedChi2)
    process.looper.monitorConfig.AlignmentMonitorSegmentDifferences.allowTIDTEC = cms.bool(allowTIDTEC)
    process.looper.monitorConfig.AlignmentMonitorSegmentDifferences.minDT13Hits = process.looper.algoConfig.minDT13Hits
    process.looper.monitorConfig.AlignmentMonitorSegmentDifferences.minDT2Hits = process.looper.algoConfig.minDT2Hits
    process.looper.monitorConfig.AlignmentMonitorSegmentDifferences.minCSCHits = process.looper.algoConfig.minCSCHits
    process.looper.monitorConfig.AlignmentMonitorSegmentDifferences.doDT = cms.bool(doDT)
    process.looper.monitorConfig.AlignmentMonitorSegmentDifferences.doCSC = cms.bool(doCSC)

if curvatureplots:
    process.load("Alignment.CommonAlignmentMonitor.AlignmentMonitorMuonVsCurvature_cfi")
    process.looper.monitorConfig.monitors.append("AlignmentMonitorMuonVsCurvature")
    process.looper.monitorConfig.AlignmentMonitorMuonVsCurvature = process.AlignmentMonitorMuonVsCurvature
    process.looper.monitorConfig.AlignmentMonitorMuonVsCurvature.minTrackerHits = cms.int32(minTrackerHits)
    process.looper.monitorConfig.AlignmentMonitorMuonVsCurvature.maxTrackerRedChi2 = cms.double(maxTrackerRedChi2)
    process.looper.monitorConfig.AlignmentMonitorMuonVsCurvature.allowTIDTEC = cms.bool(allowTIDTEC)
    process.looper.monitorConfig.AlignmentMonitorMuonVsCurvature.minDT13Hits = process.looper.algoConfig.minDT13Hits
    process.looper.monitorConfig.AlignmentMonitorMuonVsCurvature.minDT2Hits = process.looper.algoConfig.minDT2Hits
    process.looper.monitorConfig.AlignmentMonitorMuonVsCurvature.minCSCHits = process.looper.algoConfig.minCSCHits
    process.looper.monitorConfig.AlignmentMonitorMuonVsCurvature.doDT = cms.bool(doDT)
    process.looper.monitorConfig.AlignmentMonitorMuonVsCurvature.doCSC = cms.bool(doCSC)

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = cms.string(globaltag)
process.looper.applyDbAlignment = True
process.load("RecoVertex.BeamSpotProducer.BeamSpot_cfi")

if iscosmics:
    process.Path = cms.Path(process.offlineBeamSpot * process.MuonAlignmentFromReferenceGlobalCosmicRefit)
    process.looper.tjTkAssociationMapTag = cms.InputTag("MuonAlignmentFromReferenceGlobalCosmicRefit:Refitted")
else:
    process.Path = cms.Path(process.offlineBeamSpot * process.MuonAlignmentFromReferenceGlobalMuonRefit)
    process.looper.tjTkAssociationMapTag = cms.InputTag("MuonAlignmentFromReferenceGlobalMuonRefit:Refitted")

process.MuonAlignmentFromReferenceInputDB.connect = cms.string("sqlite_file:%s" % inputdb)
process.MuonAlignmentFromReferenceInputDB.toGet = cms.VPSet(cms.PSet(record = cms.string("DTAlignmentRcd"), tag = cms.string("DTAlignmentRcd")),
                                                            cms.PSet(record = cms.string("CSCAlignmentRcd"), tag = cms.string("CSCAlignmentRcd")))

if trackerconnect != "":
    from CondCore.DBCommon.CondDBSetup_cfi import *
    process.TrackerAlignmentInputDB = cms.ESSource("PoolDBESSource",
                                                   CondDBSetup,
                                                   connect = cms.string(trackerconnect),
                                                   toGet = cms.VPSet(cms.PSet(record = cms.string("TrackerAlignmentRcd"), tag = cms.string(trackeralignment))))
    process.es_prefer_TrackerAlignmentInputDB = cms.ESPrefer("PoolDBESSource", "TrackerAlignmentInputDB")

if trackerAPEconnect != "":
    from CondCore.DBCommon.CondDBSetup_cfi import *
    process.TrackerAlignmentErrorInputDB = cms.ESSource("PoolDBESSource",
                                                   CondDBSetup,
                                                   connect = cms.string(trackerAPEconnect),
                                                   toGet = cms.VPSet(cms.PSet(cms.PSet(record = cms.string("TrackerAlignmentErrorRcd"), tag = cms.string(trackerAPE)))))
    process.es_prefer_TrackerAlignmentErrorInputDB = cms.ESPrefer("PoolDBESSource", "TrackerAlignmentErrorInputDB")

if gprcdconnect != "":
    from CondCore.DBCommon.CondDBSetup_cfi import *
    process.GlobalPositionInputDB = cms.ESSource("PoolDBESSource",
                                                   CondDBSetup,
                                                   connect = cms.string(gprcdconnect),
                                                   toGet = cms.VPSet(cms.PSet(record = cms.string("GlobalPositionRcd"), tag = cms.string(gprcd))))
    process.es_prefer_GlobalPositionInputDB = cms.ESPrefer("PoolDBESSource", "GlobalPositionInputDB")


## the following was needed for Nov 2010 alignment to pick up new lorentz angle and strip conditions for tracker
#process.poolDBESSourceLA = cms.ESSource("PoolDBESSource",
#  BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
#  DBParameters = cms.PSet(
#    messageLevel = cms.untracked.int32(0),
#    authenticationPath = cms.untracked.string('.')
#    #messageLevel = cms.untracked.int32(2),
#    #authenticationPath = cms.untracked.string('/path/to/authentication')
#  ),
#  timetype = cms.untracked.string('runnumber'),
#  connect = cms.string('frontier://PromptProd/CMS_COND_31X_STRIP'),
#  toGet = cms.VPSet(cms.PSet(
#    record = cms.string('SiStripLorentzAngleRcd'),
#    tag = cms.string('SiStripLorentzAngle_GR10_v2_offline')
#  ))
#)
#process.es_prefer_LA = cms.ESPrefer('PoolDBESSource','poolDBESSourceLA')
#
#process.poolDBESSourceBP = cms.ESSource("PoolDBESSource",
#  BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
#  DBParameters = cms.PSet(
#    messageLevel = cms.untracked.int32(0),
#    authenticationPath = cms.untracked.string('.')
#    #messageLevel = cms.untracked.int32(2),
#    #authenticationPath = cms.untracked.string('/path/to/authentication')
#  ),
#  timetype = cms.untracked.string('runnumber'),
#  connect = cms.string('frontier://PromptProd/CMS_COND_31X_STRIP'),
#  toGet = cms.VPSet(cms.PSet(
#    record = cms.string('SiStripConfObjectRcd'),
#    tag = cms.string('SiStripShiftAndCrosstalk_GR10_v2_offline')
#  ))
#)
#process.es_prefer_BP = cms.ESPrefer('PoolDBESSource','poolDBESSourceBP')


process.looper.saveToDB = False
process.looper.saveApeToDB = False
del process.PoolDBOutputService

process.TFileService = cms.Service("TFileService", fileName = cms.string("plotting%03d.root" % jobnumber))
