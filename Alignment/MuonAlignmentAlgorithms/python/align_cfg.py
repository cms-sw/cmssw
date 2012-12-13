import os
import FWCore.ParameterSet.Config as cms

alignmenttmp = os.environ["ALIGNMENT_ALIGNMENTTMP"].split("\n")
iteration = int(os.environ["ALIGNMENT_ITERATION"])

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
maxResSlopeY = float(os.environ["ALIGNMENT_MAXRESSLOPEY"])
residualsModel = os.environ["ALIGNMENT_RESIDUALSMODEL"]
peakNSigma = float(os.environ["ALIGNMENT_PEAKNSIGMA"])
useResiduals = os.environ["ALIGNMENT_USERESIDUALS"]

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

# optionally: create ntuples along with tmp files
createAlignNtuple = False
envNtuple = os.getenv("ALIGNMENT_CREATEALIGNNTUPLE")
if envNtuple is not None:
  if envNtuple=='True': createAlignNtuple = True


process = cms.Process("ALIGN")
process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1))

process.load("Alignment.MuonAlignmentAlgorithms.MuonAlignmentFromReference_cff")
process.looper.algoConfig.readTemporaryFiles = cms.vstring(*alignmenttmp)
process.looper.algoConfig.reportFileName = cms.string("MuonAlignmentFromReference_report.py")

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
process.looper.algoConfig.residualsModel = cms.string(residualsModel)
process.looper.algoConfig.peakNSigma = peakNSigma
process.looper.algoConfig.createNtuple = createAlignNtuple
process.looper.algoConfig.doDT = doDT
process.looper.algoConfig.doCSC = doCSC
process.looper.algoConfig.useResiduals = cms.string(useResiduals)

#process.looper.algoConfig.specialFitPatternDT6DOF = "1000-100011_0100-010000"
process.looper.algoConfig.specialFitPatternDT6DOF = ""
process.looper.algoConfig.specialFitPatternDT5DOF = ""
process.looper.algoConfig.specialFitPatternCSC = ""

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = cms.string(globaltag)
process.looper.applyDbAlignment = True

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

if trackerBowsconnect != "":
    from CondCore.DBCommon.CondDBSetup_cfi import *
    process.TrackerSurfaceDeformationInputDB = cms.ESSource("PoolDBESSource",
                                                   CondDBSetup,
                                                   connect = cms.string(trackerBowsconnect),
                                                   toGet = cms.VPSet(cms.PSet(cms.PSet(record = cms.string("TrackerSurfaceDeformationRcd"), tag = cms.string(trackerBows)))))
    process.es_prefer_TrackerSurfaceDeformationInputDB = cms.ESPrefer("PoolDBESSource", "TrackerSurfaceDeformationInputDB")

if gprcdconnect != "":
    from CondCore.DBCommon.CondDBSetup_cfi import *
    process.GlobalPositionInputDB = cms.ESSource("PoolDBESSource",
                                                   CondDBSetup,
                                                   connect = cms.string(gprcdconnect),
                                                   toGet = cms.VPSet(cms.PSet(record = cms.string("GlobalPositionRcd"), tag = cms.string(gprcd))))
    process.es_prefer_GlobalPositionInputDB = cms.ESPrefer("PoolDBESSource", "GlobalPositionInputDB")

process.looper.saveToDB = True
process.looper.saveApeToDB = True
process.PoolDBOutputService.connect = cms.string("sqlite_file:MuonAlignmentFromReference_outputdb.db")

process.TFileService = cms.Service("TFileService", fileName = cms.string("MuonAlignmentFromReference_plotting.root"))
