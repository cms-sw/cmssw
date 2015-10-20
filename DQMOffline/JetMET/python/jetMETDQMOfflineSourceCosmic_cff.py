import FWCore.ParameterSet.Config as cms

from DQMOffline.JetMET.metDQMConfig_cff     import *
from DQMOffline.JetMET.jetAnalyzer_cff   import *
from DQMOffline.JetMET.SUSYDQMAnalyzer_cfi  import *
from RecoMET.METFilters.metFilters_cff  import *

# empty string: no correction applied
jetDQMAnalyzerAk4CaloUncleaned.runcosmics = cms.untracked.bool(True)
jetDQMAnalyzerAk4CaloUncleaned.JetCorrections = cms.InputTag("")
jetDQMAnalyzerAk4CaloUncleaned.filljetHighLevel =cms.bool(True)
 
caloMetDQMAnalyzer.runcosmics = cms.untracked.bool(True)
caloMetDQMAnalyzer.onlyCleaned = cms.untracked.bool(False)
caloMetDQMAnalyzer.JetCorrections = cms.InputTag("")


jetMETDQMOfflineSourceCosmic = cms.Sequence(HBHENoiseFilterResultProducer*AnalyzeSUSYDQM*jetDQMAnalyzerSequenceCosmics*METDQMAnalyzerSequenceCosmics)
