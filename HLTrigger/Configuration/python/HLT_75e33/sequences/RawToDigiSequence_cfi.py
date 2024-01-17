import FWCore.ParameterSet.Config as cms

from ..modules.hgcalDigis_cfi import *
from ..modules.hltHcalDigis_cfi import *
from ..modules.muonCSCDigis_cfi import *
from ..modules.muonDTDigis_cfi import *
from ..modules.muonGEMDigis_cfi import *
from ..sequences.hltEcalDigisSequence_cfi import *

RawToDigiSequence = cms.Sequence(hgcalDigis+hltEcalDigisSequence+hltHcalDigis+muonCSCDigis+muonDTDigis+muonGEMDigis)
