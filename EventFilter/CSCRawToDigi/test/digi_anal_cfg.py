import FWCore.ParameterSet.Config as cms
# packs and unpacks data from a dataset which already has digis
process = cms.Process("TEST")
process.load("CalibMuon.Configuration.CSC_FakeDBConditions_cff")
process.load("EventFilter.CSCRawToDigi.cscFrontierCablingUnpck_cff")



#process.source = cms.Source("DaqSource",
#	        readerPluginName = cms.untracked.string('CSCFileReader'),	
#                readerPset = cms.untracked.PSet( 
#                        RUI17 = cms.untracked.vstring('/tmp/barvic/csc_00062751_EmuRUI17_Monitor_000.raw'),
#                        FED750 = cms.untracked.vstring('RUI17'),
#                        firstEvent = cms.untracked.int32(0)
# 		)               
#       ) 

#process.maxEvents = cms.untracked.PSet(  input = cms.untracked.int32(2000) )

process.source = cms.Source("PoolSource",
#    skipEvents = cms.untracked.uint32(18237),
    fileNames = cms.untracked.vstring(
    'file:/tmp/barvic/statusdigi.root'
# '/store/data/Commissioning08/Cosmics/RAW/v1/000/064/257/A6B9F13F-CC90-DD11-9BCA-001617E30CE8.root'
    )
)


process.load("EventFilter.CSCRawToDigi.cscUnpacker_cfi")


process.anal = cms.EDAnalyzer("DigiAnalyzer")

process.out = cms.OutputModule("PoolOutputModule",
    fastCloning = cms.untracked.bool(False),
    fileName = cms.untracked.string('statusdigi.root')
)

process.dump = cms.EDFilter("CSCFileDumper",
    output = cms.untracked.string('out.raw')
)


process.muonCSCDigis.UnpackStatusDigis = True
process.muonCSCDigis.Debug = True
process.muonCSCDigis.UseExaminer = True

process.p = cms.Path(process.anal)
# process.p = cms.Path(process.dump)
# process.e = cms.EndPath(process.out)

