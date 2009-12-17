import FWCore.ParameterSet.Config as cms

#from JetMETCorrections.Configuration.JetPlusTrackCorrections_cff import *
#from JetMETCorrections.Configuration.ZSPJetCorrections152_cff import *

from DQMOffline.JetMET.jetMETAnalyzer_cfi import *

# no JPT :
#jetMETAnalyzerCosmicSequence = cms.Sequence(jetMETAnalyzer)
# with JPT take this sequence:
#jetMETAnalyzerSequence = cms.Sequence(ZSPJetCorrections*JetPlusTrackCorrections*jetMETAnalyzer)
# remove JPT from non-cosmic sequence too for now
jetMETAnalyzerSequence = cms.Sequence(jetMETAnalyzer)

