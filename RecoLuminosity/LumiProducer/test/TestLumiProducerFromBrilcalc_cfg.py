import FWCore.ParameterSet.Config as cms

process = cms.Process("TestLumiProducerFromBrilcalc")

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    cout = cms.untracked.PSet(
        enable = cms.untracked.bool(True),
        threshold = cms.untracked.string('DEBUG')
    ),
    debugModules = cms.untracked.vstring('LumiInfo')
)

# just use a random relval which has meaningless run/LS numbers, and then a corresponding test file
process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring("/store/relval/CMSSW_10_2_7/RelValMinBias_13/GEN-SIM/102X_mc2017_realistic_v4_AC_v01-v1/20000/BB654FAE-5375-164F-BBFE-B330713759C6.root")
)
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(10))

process.LumiInfo = cms.EDProducer('LumiProducerFromBrilcalc',
                                  lumiFile = cms.string("./testLumiFile.csv"),
                                  throwIfNotFound = cms.bool(False),
                                  doBunchByBunch = cms.bool(False))

process.test = cms.EDAnalyzer('TestLumiProducerFromBrilcalc',
                              inputTag = cms.untracked.InputTag("LumiInfo", "brilcalc"))

process.p = cms.Path(process.LumiInfo*process.test)

