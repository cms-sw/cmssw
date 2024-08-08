import FWCore.ParameterSet.Config as cms

from ..modules.hltHgcalDigis_cfi import *
from ..modules.hltHcalDigis_cfi import *
from ..modules.hltMuonCSCDigis_cfi import *
from ..modules.hltMuonDTDigis_cfi import *
from ..modules.hltMuonGEMDigis_cfi import *
from ..sequences.HLTEcalDigisSequence_cfi import *

HLTRawToDigiSequence = cms.Sequence(hltHgcalDigis+HLTEcalDigisSequence+hltHcalDigis+hltMuonCSCDigis+hltMuonDTDigis+hltMuonGEMDigis)
