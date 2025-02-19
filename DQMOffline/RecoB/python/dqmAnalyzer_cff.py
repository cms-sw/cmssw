import FWCore.ParameterSet.Config as cms

from RecoBTag.Configuration.RecoBTag_cff import *
from DQMOffline.RecoB.bTagAnalysisData_cfi import *
# Module execution
bTagPlots = cms.Sequence(bTagAnalysis)
bTagAnalysis.finalizePlots = False
bTagAnalysis.finalizeOnly = False


