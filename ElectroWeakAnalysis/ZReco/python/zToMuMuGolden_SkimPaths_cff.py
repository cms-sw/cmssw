# The following comments couldn't be translated into the new config version:

#  zToMuMuHLTFilter &

import FWCore.ParameterSet.Config as cms

from ElectroWeakAnalysis.ZReco.zToMuMuGoldenSequences_cff import *
from ElectroWeakAnalysis.ZReco.zToMuMuHLTFilter_cfi import *
import ElectroWeakAnalysis.ZReco.zToMuMuFilter_cfi
zToMuMuGoldenFilter = ElectroWeakAnalysis.ZReco.zToMuMuFilter_cfi.zToMuMuFilter.clone()
zToMuMuGoldenHLTPath = cms.Path(muonRecoForZToMuMuGolden+zToMuMuGolden+zToMuMuGoldenFilter+muonMCTruthForZToMuMuGolden)
zToMuMuGoldenFilter.src = 'zToMuMuGolden'

