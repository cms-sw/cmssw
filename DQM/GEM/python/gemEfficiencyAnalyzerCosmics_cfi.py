import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
from RecoMuon.TrackingTools.MuonServiceProxy_cff import MuonServiceProxy
from DQM.GEM.gemEfficiencyAnalyzerCosmicsDefault_cfi import gemEfficiencyAnalyzerCosmicsDefault as _gemEfficiencyAnalyzerCosmicsDefault
from TrackingTools.TransientTrack.TransientTrackBuilder_cfi import *

gemEfficiencyAnalyzerCosmics = _gemEfficiencyAnalyzerCosmicsDefault.clone(
    ServiceParameters = MuonServiceProxy.ServiceParameters.clone(),
    muonTag = 'muons',
    name = 'Cosmic 2-Leg STA Muon',
    folder = 'GEM/Efficiency/type1'
)

gemEfficiencyAnalyzerCosmicsOneLeg = _gemEfficiencyAnalyzerCosmicsDefault.clone(
    ServiceParameters = MuonServiceProxy.ServiceParameters.clone(),
    muonTag = 'muons1Leg',
    name = 'Cosmic 1-Leg STA Muon',
    folder = 'GEM/Efficiency/type2'
)

from Configuration.Eras.Modifier_phase2_GEM_cff import phase2_GEM
phase2_GEM.toModify(gemEfficiencyAnalyzerCosmics,
    etaNbins=cms.untracked.int32(15),
    etaUp=cms.untracked.double(3.0))

phase2_GEM.toModify(gemEfficiencyAnalyzerCosmicsOneLeg,
    etaNbins=cms.untracked.int32(15),
    etaUp=cms.untracked.double(3.0))
