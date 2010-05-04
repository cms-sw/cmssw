import os
import FWCore.ParameterSet.Config as cms

alignmenttmp = os.environ["ALIGNMENT_ALIGNMENTTMP"].split("\n")
iteration = int(os.environ["ALIGNMENT_ITERATION"])
mode = os.environ["ALIGNMENT_MODE"]
inputdb = os.environ["ALIGNMENT_INPUTDB"]
globaltag = os.environ["ALIGNMENT_GLOBALTAG"]

minP = float(os.environ["ALIGNMENT_minP"])
minHitsPerChamber = int(os.environ["ALIGNMENT_minHitsPerChamber"])
maxdrdz = float(os.environ["ALIGNMENT_maxdrdz"])
maxRedChi2 = float(os.environ["ALIGNMENT_maxRedChi2"])
fiducial = (os.environ["ALIGNMENT_fiducial"] == "True")
useHitWeights = (os.environ["ALIGNMENT_useHitWeights"] == "True")
combineME11 = (os.environ["ALIGNMENT_combineME11"] == "True")
useTrackWeights = (os.environ["ALIGNMENT_useTrackWeights"] == "True")
errorFromRMS = (os.environ["ALIGNMENT_errorFromRMS"] == "True")
slopeFromTrackRefit = (os.environ["ALIGNMENT_slopeFromTrackRefit"] == "True")
minStationsInTrackRefits = int(os.environ["ALIGNMENT_minStationsInTrackRefits"])

process = cms.Process("ALIGN")
process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1))

process.load("Alignment.MuonAlignmentAlgorithms.CSCOverlapsAlignmentAlgorithm_cff")
del process.Path

process.looper.algoConfig.mode = mode

process.looper.algoConfig.writeTemporaryFile = ""
process.looper.algoConfig.readTemporaryFiles = cms.vstring(*alignmenttmp)
process.looper.algoConfig.doAlignment = True

process.looper.algoConfig.minP = minP
process.looper.algoConfig.minHitsPerChamber = minHitsPerChamber
process.looper.algoConfig.maxdrdz = maxdrdz
process.looper.algoConfig.maxRedChi2 = maxRedChi2
process.looper.algoConfig.fiducial = fiducial
process.looper.algoConfig.useHitWeights = useHitWeights
process.looper.algoConfig.combineME11 = combineME11
process.looper.algoConfig.useTrackWeights = useTrackWeights
process.looper.algoConfig.errorFromRMS = errorFromRMS
process.looper.algoConfig.slopeFromTrackRefit = slopeFromTrackRefit
process.looper.algoConfig.minStationsInTrackRefits = minStationsInTrackRefits

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = cms.string(globaltag)

process.muonAlignment.connect = cms.string("sqlite_file:%s" % inputdb)

process.TFileService = cms.Service("TFileService", fileName = cms.string("plotting.root"))
