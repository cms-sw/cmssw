import FWCore.ParameterSet.Config as cms

from L1TriggerOffline.L1Analyzer.L1MuonMCAnalysis_cff import *

from L1TriggerOffline.L1Analyzer.L1EmMCAnalysis_cff import *
from L1TriggerOffline.L1Analyzer.L1IsoEmMCAnalysis_cff import *
from L1TriggerOffline.L1Analyzer.L1NonIsoEmMCAnalysis_cff import *

from L1TriggerOffline.L1Analyzer.L1JetMCAnalysis_cff import *
from L1TriggerOffline.L1Analyzer.L1CenTauJetMCAnalysis_cff import *
from L1TriggerOffline.L1Analyzer.L1ForJetMCAnalysis_cff import *
from L1TriggerOffline.L1Analyzer.L1TauJetMCAnalysis_cff import *

from L1TriggerOffline.L1Analyzer.L1MetMCAnalysis_cff import *

L1MCAnalysis = cms.Sequence(
    L1MuonMCAnalysis
    +L1EmMCAnalysis
    +L1IsoEmMCAnalysis
    +L1NonIsoEmMCAnalysis
    +L1JetMCAnalysis
    +L1CenTauJetMCAnalysis
    +L1ForJetMCAnalysis
    +L1TauJetMCAnalysis
    +L1MetMCAnalysis
)
