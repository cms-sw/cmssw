import FWCore.ParameterSet.Config as cms

process = cms.Process("LmfTest")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1000)
)

# MessageLogger:
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.limit = 0

#LMF source (lmfSource)
process.load("CalibCalorimetry.EcalLaserSorting.LmfSource_cfi")

#ECAL Data dump (dumpRaw)
#process.load("pgras.DumpRaw.DumpRaw_cfi")

#process.dumpRaw.dump = cms.untracked.bool(False) 

process.source.fileNames = [ "/localdata/disk0/craft1-sorting-new/out/EB-1/Run66079_LB0001.lmf" ]

process.Timing = cms.Service("Timing")

process.load("EventFilter.EcalRawToDigi.EcalUnpackerMapping_cfi")
process.load("EventFilter.EcalRawToDigi.EcalUnpackerData_cfi")
process.ecalEBunpacker.silentMode = cms.untracked.bool(True)
process.load("Geometry.EcalMapping.EcalMapping_cfi")
process.load("Geometry.EcalMapping.EcalMappingRecord_cfi")

#import EventFilter.ESRawToDigi.esRawToDigi_cfi

process.p = cms.Path(process.ecalEBunpacker)

