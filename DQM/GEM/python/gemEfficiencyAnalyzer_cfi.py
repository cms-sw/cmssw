import FWCore.ParameterSet.Config as cms
from DQM.GEM.gemEfficiencyAnalyzerDefault_cfi import gemEfficiencyAnalyzerDefault as _gemEfficiencyAnalyzerDefault
from RecoMuon.TrackingTools.MuonServiceProxy_cff import *
from TrackingTools.TransientTrack.TransientTrackBuilder_cfi import *

gemEfficiencyAnalyzer = _gemEfficiencyAnalyzerDefault.clone(
    ServiceParameters = MuonServiceProxy.ServiceParameters.clone()
)

from Configuration.Eras.Modifier_phase2_GEM_cff import phase2_GEM
phase2_GEM.toModify(gemEfficiencyAnalyzer,
    monitorGE21 = True,
    monitorGE0 = True,
)
