import FWCore.ParameterSet.Config as cms

from RecoBTag.Skimming.btagDijet_cfi import *
from RecoBTag.Skimming.btagDijet_HLT_cfi import *
btagDijetPath = cms.Path(btagDijet_HLT*btagDijet)

