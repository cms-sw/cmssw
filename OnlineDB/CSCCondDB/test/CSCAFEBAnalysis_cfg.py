# The following comments couldn't be translated into the new config version:

#        string TestName="AFEBThresholdScan"

import FWCore.ParameterSet.Config as cms

process = cms.Process("PROD")
process.source = cms.Source("DaqSource",
    maxEvents = cms.untracked.int32(100),
    pset = cms.PSet(
        fileNames = cms.untracked.vstring('/afs/cern.ch/user/t/teren/scratch0/Data/RunNum500Evs0to9999.bin')
    ),
    reader = cms.string('CSCFileReader')
)

process.cscunpacker = cms.EDProducer("CSCDCCUnpacker",
    Debug = cms.untracked.bool(False),
    PrintEventNumber = cms.untracked.bool(False),
    theMappingFile = cms.FileInPath('OnlineDB/CSCCondDB/test/csc_slice_test_map.txt'),
    UseExaminer = cms.untracked.bool(False)
)

process.analyzer = cms.EDAnalyzer("CSCAFEBAnalyzer",
    #        string HistogramFile = 'hist_AnodeThr30_RunNum500Evs0to9999.root'
    HistogramFile = cms.string('hist_AnodeConnect_RunNum500Evs0to9999.root'),
    TestName = cms.string('AFEBConnectivity'),
    CSCSrc = cms.InputTag("cscunpacker","MuonCSCWireDigi")
)

process.p = cms.Path(process.cscunpacker*process.analyzer)

