import FWCore.ParameterSet.Config as cms
import os,sys

sys.path = sys.path + [os.path.expandvars("$CMSSW_BASE/src/HLTrigger/Configuration/test/"), os.path.expandvars("$CMSSW_RELEASE_BASE/src/HLTrigger/Configuration/test/")]

from OnLine_HLT_GRun import process

process.hltHbherecopre = process.hltHbhereco.clone(
    makeRecHits = cms.bool(False),
    saveInfos = cms.bool(True),
)

process.hltHbhereco = cms.EDProducer("FacileHcalReconstructor",
    Client = cms.PSet(
        batchSize = cms.untracked.uint32(16000),
        address = cms.untracked.string("0.0.0.0"),
        port = cms.untracked.uint32(8001),
        timeout = cms.untracked.uint32(300),
        modelName = cms.string("facile_all_v2"),
        mode = cms.string("Async"),
        modelVersion = cms.string(""),
        verbose = cms.untracked.bool(False),
        allowedTries = cms.untracked.uint32(5),
        outputs = cms.untracked.vstring("output/BiasAdd"),
    ),
    ChannelInfoName = cms.InputTag("hltHbherecopre")
)

process.HLTDoLocalHcalSequence = cms.Sequence( process.hltHcalDigis + process.hltHbherecopre + process.hltHbhereco + process.hltHfprereco + process.hltHfreco + process.hltHoreco )
process.HLTStoppedHSCPLocalHcalReco = cms.Sequence( process.hltHcalDigis + process.hltHbherecopre + process.hltHbhereco)

from Configuration.AlCa.GlobalTag import GlobalTag as customiseGlobalTag
process.GlobalTag = customiseGlobalTag(process.GlobalTag, globaltag = '112X_mcRun3_2021_realistic_v7')

process.source.fileNames = cms.untracked.vstring("root://cmsxrootd.fnal.gov//store/relval/CMSSW_11_2_0_pre6_ROOT622/RelValTTbar_14TeV/GEN-SIM-DIGI-RAW/112X_mcRun3_2021_realistic_v7-v1/20000/FED4709C-569E-0A42-8FF7-20E565ABE999.root")
