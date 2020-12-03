import FWCore.ParameterSet.Config as cms
import os,sys

sys.path = sys.path + [os.path.expandvars("$CMSSW_BASE/src/HLTrigger/Configuration/test/"), os.path.expandvars("$CMSSW_RELEASE_BASE/src/HLTrigger/Configuration/test/")]

from OnLine_HLT_GRun import process

process.hltHbherecopre = process.hltHbhereco.clone(
    makeRecHits = cms.bool(False),
    saveInfos = cms.bool(True),
)

from RecoLocalCalo.HcalRecProducers.facileHcalReconstructor_cfi import sonic_hbheprereco
process.hltHbhereco = sonic_hbheprereco.clone(
    ChannelInfoName = cms.InputTag("hltHbherecopre")
)

process.HLTDoLocalHcalSequence = cms.Sequence( process.hltHcalDigis + process.hltHbherecopre + process.hltHbhereco + process.hltHfprereco + process.hltHfreco + process.hltHoreco )
process.HLTStoppedHSCPLocalHcalReco = cms.Sequence( process.hltHcalDigis + process.hltHbherecopre + process.hltHbhereco)

from Configuration.AlCa.GlobalTag import GlobalTag as customiseGlobalTag
process.GlobalTag = customiseGlobalTag(process.GlobalTag, globaltag = '112X_mcRun3_2021_realistic_v11')

process.source.fileNames = cms.untracked.vstring("/store/relval/CMSSW_11_2_0_pre6_ROOT622/RelValTTbar_14TeV/GEN-SIM-DIGI-RAW/112X_mcRun3_2021_realistic_v7-v1/20000/FED4709C-569E-0A42-8FF7-20E565ABE999.root")
