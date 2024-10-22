import FWCore.ParameterSet.Config as cms

# the uGT DQM modules
from DQM.L1TMonitor.L1TStage2uGT_cfi import *
from DQM.L1TMonitor.L1TStage2uGTTiming_cfi import *

# Calo L2 output to uGT input comparison
from DQM.L1TMonitor.L1TStage2uGTCaloLayer2Comp_cfi import *

# uGT Board Comparison
from DQM.L1TMonitor.L1TStage2uGTBoardComp_cff import *

# compares the unpacked uGMT muon collection to the unpacked uGT muon collection
# only muons that do not match are filled in the histograms
from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
l1tStage2uGMTOutVsuGTIn = DQMEDAnalyzer(
    "L1TStage2MuonComp",
    muonCollection1 = cms.InputTag("gmtStage2Digis", "Muon"),
    muonCollection2 = cms.InputTag("gtStage2Digis", "Muon"),
    monitorDir = cms.untracked.string("L1T/L1TStage2uGT/uGMToutput_vs_uGTinput"),
    muonCollection1Title = cms.untracked.string("uGMT output muons"),
    muonCollection2Title = cms.untracked.string("uGT input muons"),
    summaryTitle = cms.untracked.string("Summary of comparison between uGMT output muons and uGT input muons"),
    ignoreBin = cms.untracked.vint32([1]), # ignore the BX range bin
    verbose = cms.untracked.bool(False),
)

# sequences
l1tStage2uGTOnlineDQMSeq = cms.Sequence(
    l1tStage2uGT +
    l1tStage2uGTCaloLayer2Comp +
    l1tStage2uGMTOutVsuGTIn
)

_run3_l1tStage2uGTOnlineDQMSeq = cms.Sequence(
    l1tStage2uGT +
    l1tStage2uGTTiming +
    l1tStage2uGTCaloLayer2Comp +
    l1tStage2uGMTOutVsuGTIn
)

# validation sequence
l1tStage2uGTValidationEventOnlineDQMSeq = cms.Sequence(
    l1tStage2uGTBoardCompSeq
)


from Configuration.Eras.Modifier_run3_common_cff import run3_common
run3_common.toReplaceWith(l1tStage2uGTOnlineDQMSeq, _run3_l1tStage2uGTOnlineDQMSeq)
