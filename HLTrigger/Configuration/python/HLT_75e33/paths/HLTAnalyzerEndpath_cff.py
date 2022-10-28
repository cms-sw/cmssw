import FWCore.ParameterSet.Config as cms

from ..modules.hltPreHLTAnalyzerEndpath_cfi import *
from ..modules.hltTrigReport_cfi import *

HLTAnalyzerEndpath = cms.EndPath(
    hltPreHLTAnalyzerEndpath + 
    hltTrigReport )
