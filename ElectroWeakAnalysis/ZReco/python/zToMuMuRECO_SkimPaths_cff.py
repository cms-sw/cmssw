import FWCore.ParameterSet.Config as cms

from ElectroWeakAnalysis.ZReco.zToMuMuSequences_cff import *
from ElectroWeakAnalysis.ZReco.zToMuMuRECOHLTFilter_cfi import *
from ElectroWeakAnalysis.ZReco.zToMuMuFilter_cfi import *
from ElectroWeakAnalysis.ZReco.zToMuMuOneTrackFilter_cfi import *
from ElectroWeakAnalysis.ZReco.zToMuMuOneStandAloneMuonTrackFilter_cfi import *
zToMuMuRECOPath = cms.Path(zToMuMuRECOHLTFilter+cms.SequencePlaceholder("muonRecoForZToMuMu")+cms.SequencePlaceholder("zToMuMu")+zToMuMuFilter)
zToMuMuOneTrackRECOPath = cms.Path(zToMuMuRECOHLTFilter+cms.SequencePlaceholder("muonRecoForZToMuMu")+cms.SequencePlaceholder("zToMuMuOneTrack")+zToMuMuOneTrackFilter)
zToMuMuOneStandAloneMuonTrackRECOPath = cms.Path(zToMuMuRECOHLTFilter+cms.SequencePlaceholder("muonRecoForZToMuMu")+cms.SequencePlaceholder("zToMuMuOneStandAloneMuonTrack")+zToMuMuOneStandAloneMuonTrackFilter)

