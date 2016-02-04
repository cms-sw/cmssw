
import FWCore.ParameterSet.Config as cms

process = cms.Process("Test")
# "old-style" ECAL scale
process.load("CalibCalorimetry.EcalTPGTools.ecalTPGScale_cff")
process.EcalTrigPrimESProducer.DatabaseFile = cms.untracked.string('TPG_startup.txt.gz')

# "old-style" HCAL scale
process.load("SimCalorimetry.HcalTrigPrimProducers.hcaltpdigi_cff")

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

#keep the logging output to a nice level
process.MessageLogger = cms.Service("MessageLogger")

process.write = cms.EDAnalyzer("L1CaloInputScalesGenerator")

process.p = cms.Path(process.write)


