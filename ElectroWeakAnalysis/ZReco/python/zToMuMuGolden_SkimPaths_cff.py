# The following comments couldn't be translated into the new config version:

#  zToMuMuHLTFilter &

import FWCore.ParameterSet.Config as cms

from ElectroWeakAnalysis.ZReco.zToMuMuGoldenSequences_cff import *
from ElectroWeakAnalysis.ZReco.zToMuMuHLTFilter_cfi import *
import copy
from ElectroWeakAnalysis.ZReco.zToMuMuFilter_cfi import *
zToMuMuGoldenFilter = copy.deepcopy(zToMuMuFilter)
zToMuMuGoldenHLTPath = cms.Path(muonRecoForZToMuMuGolden+zToMuMuGolden+zToMuMuGoldenFilter+muonMCTruthForZToMuMuGolden)
zToMuMuGoldenFilter.src = 'zToMuMuGolden'

