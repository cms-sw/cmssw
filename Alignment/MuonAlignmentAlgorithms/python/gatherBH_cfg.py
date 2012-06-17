import os
import FWCore.ParameterSet.Config as cms

inputfiles = os.environ["ALIGNMENT_INPUTFILES"].split(" ")
iteration = int(os.environ["ALIGNMENT_ITERATION"])
jobnumber = int(os.environ["ALIGNMENT_JOBNUMBER"])
mode = os.environ["ALIGNMENT_MODE"]
inputdb = os.environ["ALIGNMENT_INPUTDB"]
globaltag = os.environ["ALIGNMENT_GLOBALTAG"]
photogrammetry = (os.environ["ALIGNMENT_PHOTOGRAMMETRY"] == "True")
disks = (os.environ["ALIGNMENT_DISKS"] == "True")

minP = float(os.environ["ALIGNMENT_minP"])
minHitsPerChamber = int(os.environ["ALIGNMENT_minHitsPerChamber"])
maxdrdz = float(os.environ["ALIGNMENT_maxdrdz"])
maxRedChi2 = float(os.environ["ALIGNMENT_maxRedChi2"])
fiducial = (os.environ["ALIGNMENT_fiducial"] == "True")
useHitWeights = (os.environ["ALIGNMENT_useHitWeights"] == "True")
truncateSlopeResid = float(os.environ["ALIGNMENT_truncateSlopeResid"])
truncateOffsetResid = float(os.environ["ALIGNMENT_truncateOffsetResid"])
combineME11 = (os.environ["ALIGNMENT_combineME11"] == "True")
useTrackWeights = (os.environ["ALIGNMENT_useTrackWeights"] == "True")
errorFromRMS = (os.environ["ALIGNMENT_errorFromRMS"] == "True")
minTracksPerOverlap = int(os.environ["ALIGNMENT_minTracksPerOverlap"])
slopeFromTrackRefit = (os.environ["ALIGNMENT_slopeFromTrackRefit"] == "True")
minStationsInTrackRefits = int(os.environ["ALIGNMENT_minStationsInTrackRefits"])

process = cms.Process("GATHER")
process.source = cms.Source("PoolSource", fileNames = cms.untracked.vstring(*inputfiles))
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(-1))

process.MessageLogger = cms.Service("MessageLogger",
                                    destinations = cms.untracked.vstring("cout"),
                                    cout = cms.untracked.PSet(threshold = cms.untracked.string("ERROR")))

process.load("Alignment.MuonAlignmentAlgorithms.CSCOverlapsAlignmentAlgorithm_cff")

process.looper.algoConfig.mode = mode

if disks:
    import Alignment.MuonAlignmentAlgorithms.CSCOverlapsAlignmentAlgorithm_diskfitters_cff
    process.looper.algoConfig.fitters = Alignment.MuonAlignmentAlgorithms.CSCOverlapsAlignmentAlgorithm_diskfitters_cff.fitters

execfile("constraints_cff.py")

if photogrammetry and mode != "phipos":
    for f in process.looper.algoConfig.fitters:
        if "PGFrame" in f.alignables:
            f.fixed = cms.string("PGFrame")

process.looper.algoConfig.writeTemporaryFile = "alignment%03d.tmp" % jobnumber
process.looper.algoConfig.readTemporaryFiles = cms.vstring()
process.looper.algoConfig.doAlignment = False

process.looper.algoConfig.minP = minP
process.looper.algoConfig.minHitsPerChamber = minHitsPerChamber
process.looper.algoConfig.maxdrdz = maxdrdz
process.looper.algoConfig.maxRedChi2 = maxRedChi2
process.looper.algoConfig.fiducial = fiducial
process.looper.algoConfig.useHitWeights = useHitWeights
process.looper.algoConfig.truncateSlopeResid = truncateSlopeResid
process.looper.algoConfig.truncateOffsetResid = truncateOffsetResid
process.looper.algoConfig.combineME11 = combineME11
process.looper.algoConfig.useTrackWeights = useTrackWeights
process.looper.algoConfig.errorFromRMS = errorFromRMS
process.looper.algoConfig.minTracksPerOverlap = minTracksPerOverlap
process.looper.algoConfig.slopeFromTrackRefit = slopeFromTrackRefit
process.looper.algoConfig.minStationsInTrackRefits = minStationsInTrackRefits

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = cms.string(globaltag)

process.muonAlignment.connect = cms.string("sqlite_file:%s" % inputdb)

process.looper.saveToDB = False
process.looper.saveApeToDB = False
del process.PoolDBOutputService

process.TFileService = cms.Service("TFileService", fileName = cms.string("plotting%03d.root" % jobnumber))
