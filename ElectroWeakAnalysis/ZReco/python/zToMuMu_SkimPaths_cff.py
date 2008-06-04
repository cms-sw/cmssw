# The following comments couldn't be translated into the new config version:

#  zToMuMuHLTFilter &

#  zToMuMuHLTFilter &

#  zToMuMuHLTFilter &

#  zToMuMuHLTFilter &

import FWCore.ParameterSet.Config as cms

from ElectroWeakAnalysis.ZReco.zToMuMuSequences_cff import *
from ElectroWeakAnalysis.ZReco.zToMuMuHLTFilter_cfi import *
from ElectroWeakAnalysis.ZReco.goodZToMuMuFilter_cfi import *
from ElectroWeakAnalysis.ZReco.goodZToMuMuOneTrackFilter_cfi import *
from ElectroWeakAnalysis.ZReco.goodZToMuMuOneStandAloneMuonTrackFilter_cfi import *
zToMuMuPath = cms.Path(goodMuonRecoForZToMuMu+goodZToMuMu+goodZToMuMuFilter)
zToMuMuOneTrackPath = cms.Path(goodMuonRecoForZToMuMu+goodZToMuMuOneTrack+goodZToMuMuOneTrackFilter)
zToMuMuOneStandAloneMuonTrackPath = cms.Path(goodMuonRecoForZToMuMu+goodZToMuMuOneStandAloneMuonTrack+goodZToMuMuOneStandAloneMuonTrackFilter)
zToMuMuMCTruth = cms.Path(mcTruthForZToMuMu+mcTruthForZToMuMuOneTrack+mcTruthForZToMuMuOneStandAloneMuonTrack+goodZMCMatch)

