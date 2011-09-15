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
gprcdconnect = os.environ["ALIGNMENT_GPRCDCONNECT"]
gprcd = os.environ["ALIGNMENT_GPRCD"]
iscosmics = (os.environ["ALIGNMENT_ISCOSMICS"] == "True")
station123params = os.environ["ALIGNMENT_STATION123PARAMS"]
station4params = os.environ["ALIGNMENT_STATION4PARAMS"]
cscparams = os.environ["ALIGNMENT_CSCPARAMS"]
minTrackPt = float(os.environ["ALIGNMENT_MINTRACKPT"])
maxTrackPt = float(os.environ["ALIGNMENT_MAXTRACKPT"])
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

process = cms.Process("GATHER")
process.source = cms.Source("PoolSource", fileNames = cms.untracked.vstring(*inputfiles), skipEvents = cms.untracked.uint32(skipEvents))
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(maxEvents))

process.MessageLogger = cms.Service("MessageLogger",
                                    destinations = cms.untracked.vstring("cout"),
                                    cout = cms.untracked.PSet(threshold = cms.untracked.string("ERROR")))

process.load("Alignment.MuonAlignmentAlgorithms.MuonAlignmentFromReference_cff")
process.looper.algoConfig.writeTemporaryFile = "alignment%03d.tmp" % jobnumber
process.looper.algoConfig.doAlignment = False

process.looper.ParameterBuilder.Selector.alignParams = cms.vstring("MuonDTChambers,%s,stations123" % station123params, "MuonDTChambers,%s,station4" % station4params, "MuonCSCChambers,%s" % cscparams)
process.looper.algoConfig.minTrackPt = minTrackPt
process.looper.algoConfig.maxTrackPt = maxTrackPt
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

if segdiffplots:
    process.load("Alignment.CommonAlignmentMonitor.AlignmentMonitorSegmentDifferences_cfi")
    process.looper.monitorConfig.monitors.append("AlignmentMonitorSegmentDifferences")
    process.looper.monitorConfig.AlignmentMonitorSegmentDifferences = process.AlignmentMonitorSegmentDifferences
    # this should be set separately (50 GeV nominal)
    # process.looper.monitorConfig.AlignmentMonitorSegmentDifferences.minTrackPt = cms.double(minTrackPt)
    process.looper.monitorConfig.AlignmentMonitorSegmentDifferences.minTrackerHits = cms.int32(minTrackerHits)
    process.looper.monitorConfig.AlignmentMonitorSegmentDifferences.maxTrackerRedChi2 = cms.double(maxTrackerRedChi2)
    process.looper.monitorConfig.AlignmentMonitorSegmentDifferences.allowTIDTEC = cms.bool(allowTIDTEC)
    process.looper.monitorConfig.AlignmentMonitorSegmentDifferences.minDT13Hits = process.looper.algoConfig.minDT13Hits
    process.looper.monitorConfig.AlignmentMonitorSegmentDifferences.minDT2Hits = process.looper.algoConfig.minDT2Hits
    process.looper.monitorConfig.AlignmentMonitorSegmentDifferences.minCSCHits = process.looper.algoConfig.minCSCHits

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

process.looper.saveToDB = False
process.looper.saveApeToDB = False
del process.PoolDBOutputService

process.TFileService = cms.Service("TFileService", fileName = cms.string("plotting%03d.root" % jobnumber))
