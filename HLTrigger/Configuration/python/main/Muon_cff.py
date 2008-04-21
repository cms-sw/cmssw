import FWCore.ParameterSet.Config as cms

from HLTrigger.Muon.PathSingleMu_1032_Iso_cff import *
from HLTrigger.Muon.PathSingleMu_1032_NoIso_cff import *
from HLTrigger.Muon.PathDiMuon_1032_Iso_cff import *
from HLTrigger.Muon.PathDiMuon_1032_NoIso_cff import *
from HLTrigger.Muon.PathJpsimumu_cff import *
from HLTrigger.Muon.PathUpsilonmumu_cff import *
from HLTrigger.Muon.PathZmumu_cff import *
from HLTrigger.Muon.PathMultiMuon_1032_NoIso_cff import *
from HLTrigger.Muon.PathSameSignMu_cff import *
from HLTrigger.Muon.PathSingleMu_1032_Prescale3_cff import *
from HLTrigger.Muon.PathSingleMu_1032_Prescale5_cff import *
from HLTrigger.Muon.PathSingleMu_1032_Prescale7_7_cff import *
from HLTrigger.Muon.PathSingleMu_1032_Prescale7_10_cff import *
from HLTrigger.Muon.PathMuLevel1_1032_cff import *
from HLTrigger.Muon.PathSingleMu_1032_RelaxedVtx2cm_cff import *
from HLTrigger.Muon.PathSingleMu_1032_RelaxedVtx2mm_cff import *
from HLTrigger.Muon.PathDiMuon_1032_RelaxedVtx2cm_cff import *
from HLTrigger.Muon.PathDiMuon_1032_RelaxedVtx2mm_cff import *
HLT1MuonIso = cms.Path(singleMuIso+cms.SequencePlaceholder("hltEnd"))
HLT1MuonNonIso = cms.Path(singleMuNoIso+cms.SequencePlaceholder("hltEnd"))
HLT2MuonIso = cms.Path(diMuonIso+cms.SequencePlaceholder("hltEnd"))
HLT2MuonNonIso = cms.Path(diMuonNoIso+cms.SequencePlaceholder("hltEnd"))
HLT2MuonJPsi = cms.Path(jpsiMM+cms.SequencePlaceholder("hltEnd"))
HLT2MuonUpsilon = cms.Path(upsilonMM+cms.SequencePlaceholder("hltEnd"))
HLT2MuonZ = cms.Path(zMM+cms.SequencePlaceholder("hltEnd"))
HLTNMuonNonIso = cms.Path(multiMuonNoIso+cms.SequencePlaceholder("hltEnd"))
HLT2MuonSameSign = cms.Path(sameSignMu+cms.SequencePlaceholder("hltEnd"))
HLT1MuonPrescalePt3 = cms.Path(singleMuPrescale3+cms.SequencePlaceholder("hltEnd"))
HLT1MuonPrescalePt5 = cms.Path(singleMuPrescale5+cms.SequencePlaceholder("hltEnd"))
HLT1MuonPrescalePt7x7 = cms.Path(singleMuPrescale77+cms.SequencePlaceholder("hltEnd"))
HLT1MuonPrescalePt7x10 = cms.Path(singleMuPrescale710+cms.SequencePlaceholder("hltEnd"))
HLT1MuonLevel1 = cms.Path(muLevel1Path+cms.SequencePlaceholder("hltEnd"))
CandHLT1MuonPrescaleVtx2cm = cms.Path(singleMuNoIsoRelaxedVtx2cm+cms.SequencePlaceholder("hltEnd"))
CandHLT1MuonPrescaleVtx2mm = cms.Path(singleMuNoIsoRelaxedVtx2mm+cms.SequencePlaceholder("hltEnd"))
CandHLT2MuonPrescaleVtx2cm = cms.Path(diMuonNoIsoRelaxedVtx2cm+cms.SequencePlaceholder("hltEnd"))
CandHLT2MuonPrescaleVtx2mm = cms.Path(diMuonNoIsoRelaxedVtx2mm+cms.SequencePlaceholder("hltEnd"))

