import FWCore.ParameterSet.Config as cms

process = cms.Process("EcalLaserSorting")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)


#source reading continuously files as they arrive in in/ directory:
process.load("CalibCalorimetry.EcalLaserSorting.watcherSource_cfi")

# MessageLogger:
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.limit = 0

#Event sorting (laserSorter)
process.load("CalibCalorimetry.EcalLaserSorting.laserSorter_cfi")

#process.laserSorter.disableOutput = cms.untracked.bool(True)

process.p = cms.Path(process.laserSorter)
