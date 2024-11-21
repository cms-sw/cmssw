import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer

unprescaledAlgoList = cms.untracked.vstring(
    "L1_SingleMu22_BMTF",
    "L1_SingleMu22_OMTF",
    "L1_SingleMu22_EMTF",
    "L1_SingleIsoEG28er1p5",
    "L1_SingleIsoEG32er2p5",
    "L1_SingleEG40er2p5",
    "L1_SingleEG60",
    "L1_SingleTau120er2p1",
    "L1_SingleJet180",
    "L1_ETMHF130",
    "L1_HTT360er",
    "L1_ETT2000"
)
prescaledAlgoList = cms.untracked.vstring(
    "L1_FirstCollisionInTrain",
    "L1_LastCollisionInTrain",
    "L1_IsolatedBunch",
    "L1_SingleMu0_BMTF",
    "L1_SingleMu0_OMTF",
    "L1_SingleMu0_EMTF",
    "L1_SingleEG10er2p5",
    "L1_SingleEG15er2p5",
    "L1_SingleEG26er2p5",
    "L1_SingleLooseIsoEG28er1p5",
    "L1_SingleJet35",
    "L1_SingleJet35er2p5",
    "L1_SingleJet35_FWD2p5",
    "L1_ETMHF100",
    "L1_HTT120er",
    "L1_ETT1600"
)

unprescaledAlgoList_2024 = cms.untracked.vstring(unprescaledAlgoList)
unprescaledAlgoList_2024.extend([
    "L1_AXO_Nominal",
    "L1_AXO_VTight",
    "L1_CICADA_Medium",
    "L1_CICADA_VTight"
])

unprescaledAlgoList_PbPb = cms.untracked.vstring(unprescaledAlgoList)
unprescaledAlgoList_PbPb.remove("L1_SingleIsoEG28er1p5")
unprescaledAlgoList_PbPb.remove("L1_SingleTau120er2p1")
unprescaledAlgoList_PbPb.remove("L1_ETMHF130")

prescaledAlgoList_2024 = cms.untracked.vstring(prescaledAlgoList)
if "L1_ETT1600" in prescaledAlgoList_2024:
    prescaledAlgoList_2024.remove("L1_ETT1600")

prescaledAlgoList_PbPb = cms.untracked.vstring(prescaledAlgoList)
prescaledAlgoList_PbPb.remove("L1_SingleLooseIsoEG28er1p5")
prescaledAlgoList_PbPb.remove("L1_SingleJet35_FWD2p5")
prescaledAlgoList_PbPb.remove("L1_ETT1600")

l1tStage2uGTTiming = DQMEDAnalyzer('L1TStage2uGTTiming',
    l1tStage2uGtSource = cms.InputTag("gtStage2Digis"),
    monitorDir = cms.untracked.string("L1T/L1TStage2uGT/timing_aux"),
    verbose = cms.untracked.bool(False),
    firstBXInTrainAlgo = cms.untracked.string("L1_FirstCollisionInTrain"),
    lastBXInTrainAlgo = cms.untracked.string("L1_LastCollisionInTrain"),
    isoBXAlgo = cms.untracked.string("L1_IsolatedBunch"),
    unprescaledAlgoShortList = unprescaledAlgoList,
    prescaledAlgoShortList = prescaledAlgoList,
    useAlgoDecision = cms.untracked.string("initial")
)

from Configuration.Eras.Modifier_stage2L1Trigger_2024_cff import stage2L1Trigger_2024
stage2L1Trigger_2024.toModify(l1tStage2uGTTiming,
    unprescaledAlgoShortList = unprescaledAlgoList_2024,
    prescaledAlgoShortList = prescaledAlgoList_2024
)

from Configuration.Eras.Modifier_pp_on_PbPb_run3_cff import pp_on_PbPb_run3
from Configuration.Eras.Modifier_run3_upc_cff import run3_upc
(pp_on_PbPb_run3 | run3_upc).toModify(l1tStage2uGTTiming,
                                      unprescaledAlgoShortList = unprescaledAlgoList_PbPb,
                                      prescaledAlgoShortList = prescaledAlgoList_PbPb)
