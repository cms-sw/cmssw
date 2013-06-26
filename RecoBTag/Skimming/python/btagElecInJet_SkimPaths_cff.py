import FWCore.ParameterSet.Config as cms

from RecoBTag.Skimming.btagElecInJet_HLT_cfi import *
from RecoBTag.Skimming.btagElecInJet_cfi import *
btagElecInJetPath = cms.Path(btagElecInJet_HLT*btagElecInJet)

