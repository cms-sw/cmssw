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
        modelVersion = cms.string(-1),
        verbose = cms.untracked.bool(False),
        allowedTries = cms.untracked.uint32(5),
        outputs = cms.untracked.vstring("output/BiasAdd"),
    ),
    ChannelInfoName = cms.InputTag("hltHbherecopre")
)

process.HLTDoLocalHcalSequence = cms.Sequence( process.hltHcalDigis + process.hltHbherecopre + process.hltHbhereco + process.hltHfprereco + process.hltHfreco + process.hltHoreco )
process.HLTStoppedHSCPLocalHcalReco = cms.Sequence( process.hltHcalDigis + process.hltHbherecopre + process.hltHbhereco)

process.source.fileNames = cms.untracked.vstring("file:RelVal_Raw_GRun_MC.root")
