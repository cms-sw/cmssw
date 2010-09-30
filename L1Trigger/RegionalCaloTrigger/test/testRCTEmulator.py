import FWCore.ParameterSet.Config as cms

process = cms.Process("L1")
process.load("FWCore.MessageLogger.MessageLogger_cfi")

#process.load("DQMServices.Core.DQM_cfg")
process.DQMStore = cms.Service("DQMStore")



process.source = cms.Source(
    "PoolSource",
    fileNames = cms.untracked.vstring(
#    '/store/data/Run2010B/TestEnables/RAW/v1/000/146/150/98DEA325-BBC3-DF11-9EDD-0030487C90EE.root',
    '/store/data/Run2010B/TestEnables/RAW/v1/000/146/153/049306C1-CEC3-DF11-BC34-0030487CD7C6.root',
        
        
    )
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

# standard includes
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'GR10_P_V9::All'


# unpack raw data
process.load("Configuration.StandardSequences.RawToDigi_Data_cff")


process.rctEmulDigis = cms.EDProducer("L1RCTProducer",
                                  hcalDigis = cms.VInputTag(cms.InputTag("hcalTriggerPrimitiveDigis")),
                                  useDebugTpgScales = cms.bool(True),
                                  useEcal = cms.bool(True),
                                  useHcal = cms.bool(True),
                                  ecalDigis = cms.VInputTag(cms.InputTag("ecalTriggerPrimitiveDigis")),
                                  BunchCrossings = cms.vint32(0)
                              )

process.rctEmulDigis.hcalDigis = cms.VInputTag(cms.InputTag("hcalDigis"))
process.rctEmulDigis.ecalDigis = cms.VInputTag(cms.InputTag("ecalDigis:EcalTriggerPrimitives"))

process.l1tderct = cms.EDAnalyzer("L1TdeRCT",
                              rctSourceData = cms.InputTag("gctDigis"),
                              HistFolder = cms.untracked.string('L1TEMU/L1TdeRCT/'),
                              outputFile = cms.untracked.string('dqm.root'),
                              verbose = cms.untracked.bool(False),
                              DQMStore = cms.untracked.bool(True),
                              singlechannelhistos = cms.untracked.bool(False),
                              ecalTPGData = cms.InputTag("ecalDigis","EcalTriggerPrimitives"),
                              rctSourceEmul = cms.InputTag("valRctDigis"),
                              disableROOToutput = cms.untracked.bool(False),
                              hcalTPGData = cms.InputTag("hcalDigis"),
                              gtDigisLabel = cms.InputTag("gtDigis"),
                              gtEGAlgoName = cms.string("L1_SingleEG1"),
                              doubleThreshold = cms.int32(3),

                          )


process.l1tderct.rctSourceEmul = 'rctEmulDigis'


process.p = cms.Path(
    process.RawToDigi*
    process.rctEmulDigis*
    process.l1tderct
)


