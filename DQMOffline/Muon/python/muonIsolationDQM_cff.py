import FWCore.ParameterSet.Config as cms

from RecoMuon.MuonIsolationProducers.muIsoDeposits_cff import *
from TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAny_cfi import *

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
MuIsoDQM_trk = DQMEDAnalyzer('MuonIsolationDQM',
                              Global_Muon_Label = cms.untracked.InputTag("muons"),
                              requireTRKMuon = cms.untracked.bool(True),
                              requireSTAMuon = cms.untracked.bool(False),
                              requireGLBMuon = cms.untracked.bool(False),                        
                              ecalIsoDeposit_Label = cms.untracked.InputTag("muIsoDepositCalByAssociatorTowers","ecal"),
                              #    rootfilename = cms.untracked.string('ttbar-DQMidation.root'),
                              hcalIsoDeposit_Label = cms.untracked.InputTag("muIsoDepositCalByAssociatorTowers","hcal"),
                              tkIsoDeposit_Label = cms.untracked.InputTag("muIsoDepositTk"),
                              hoIsoDeposit_Label = cms.untracked.InputTag("muIsoDepositCalByAssociatorTowers","ho"),
                              vertexLabel = cms.untracked.InputTag("offlinePrimaryVertices"),
                              directory = cms.string("Muons/Isolation/tracker"),
                             vtxBin = cms.int32(30),
                             vtxMax = cms.double(149.5),
                             vtxMin = cms.double(0.5)                             
                              )

MuIsoDQM_sta = DQMEDAnalyzer('MuonIsolationDQM',
                              Global_Muon_Label = cms.untracked.InputTag("muons"),
                              requireTRKMuon = cms.untracked.bool(False),
                              requireSTAMuon = cms.untracked.bool(True),
                              requireGLBMuon = cms.untracked.bool(False),
                              ecalIsoDeposit_Label = cms.untracked.InputTag("muIsoDepositCalByAssociatorTowers","ecal"),
                              #    rootfilename = cms.untracked.string('ttbar-DQMidation.root'),
                              hcalIsoDeposit_Label = cms.untracked.InputTag("muIsoDepositCalByAssociatorTowers","hcal"),
                              tkIsoDeposit_Label = cms.untracked.InputTag("muIsoDepositTk"),
                              hoIsoDeposit_Label = cms.untracked.InputTag("muIsoDepositCalByAssociatorTowers","ho"),
                              vertexLabel = cms.untracked.InputTag("offlinePrimaryVertices"),
                              directory = cms.string("Muons/Isolation/standalone"),
                             vtxBin = cms.int32(30),
                             vtxMax = cms.double(149.5),
                             vtxMin = cms.double(0.5)
                              )

MuIsoDQM_glb = DQMEDAnalyzer('MuonIsolationDQM',
                              Global_Muon_Label = cms.untracked.InputTag("muons"),
                              requireTRKMuon = cms.untracked.bool(False),
                              requireSTAMuon = cms.untracked.bool(False),
                              requireGLBMuon = cms.untracked.bool(True),
                              ecalIsoDeposit_Label = cms.untracked.InputTag("muIsoDepositCalByAssociatorTowers","ecal"),
                              #    rootfilename = cms.untracked.string('ttbar-DQMidation.root'),
                              hcalIsoDeposit_Label = cms.untracked.InputTag("muIsoDepositCalByAssociatorTowers","hcal"),
                              tkIsoDeposit_Label = cms.untracked.InputTag("muIsoDepositTk"),
                              hoIsoDeposit_Label = cms.untracked.InputTag("muIsoDepositCalByAssociatorTowers","ho"),
                              vertexLabel = cms.untracked.InputTag("offlinePrimaryVertices"),
                              directory = cms.string("Muons/Isolation/global"),
                             vtxBin = cms.int32(30),
                             vtxMax = cms.double(149.5),
                             vtxMin = cms.double(0.5)
                              )
muIsoDQM_seq = cms.Sequence(MuIsoDQM_trk+MuIsoDQM_sta+MuIsoDQM_glb)

MuIsoDQM_glb_Phase2=MuIsoDQM_glb.clone(
    vtxBin=20,
    vtxMin=149.5,
    vtxMax=249.5
)

MuIsoDQM_trk_Phase2=MuIsoDQM_trk.clone(
    vtxBin=20,
    vtxMin=149.5,
    vtxMax=249.5                                                                                                    
)

MuIsoDQM_sta_Phase2=MuIsoDQM_sta.clone(
    vtxBin=20,
    vtxMin=149.5,
    vtxMax=249.5                                                                                                    
)

muIsoDQM_seq_Phase2 = cms.Sequence(MuIsoDQM_trk_Phase2+MuIsoDQM_sta_Phase2+MuIsoDQM_glb_Phase2) 

MuIsoDQM_trk_miniAOD = DQMEDAnalyzer('MuonIsolationDQM',
                                      Global_Muon_Label = cms.untracked.InputTag("slimmedMuons"),
                                      requireTRKMuon = cms.untracked.bool(True),
                                      requireSTAMuon = cms.untracked.bool(False),
                                      requireGLBMuon = cms.untracked.bool(False),                        
                                      ecalIsoDeposit_Label = cms.untracked.InputTag("muIsoDepositCalByAssociatorTowers","ecal"),
                                      #    rootfilename = cms.untracked.string('ttbar-DQMidation.root'),
                                      hcalIsoDeposit_Label = cms.untracked.InputTag("muIsoDepositCalByAssociatorTowers","hcal"),
                                      tkIsoDeposit_Label = cms.untracked.InputTag("muIsoDepositTk"),
                                      hoIsoDeposit_Label = cms.untracked.InputTag("muIsoDepositCalByAssociatorTowers","ho"),
                                      vertexLabel = cms.untracked.InputTag("offlinePrimaryVertices"),
                                      directory = cms.string("Muons_miniAOD/Isolation/tracker"),
                                     vtxBin = cms.int32(30),
                                     vtxMax = cms.double(149.5),
                                     vtxMin = cms.double(0.5)                             
                                      )

MuIsoDQM_sta_miniAOD = DQMEDAnalyzer('MuonIsolationDQM',
                                      Global_Muon_Label = cms.untracked.InputTag("slimmedMuons"),
                                      requireTRKMuon = cms.untracked.bool(False),
                                      requireSTAMuon = cms.untracked.bool(True),
                                      requireGLBMuon = cms.untracked.bool(False),
                                      ecalIsoDeposit_Label = cms.untracked.InputTag("muIsoDepositCalByAssociatorTowers","ecal"),
                                      #    rootfilename = cms.untracked.string('ttbar-DQMidation.root'),
                                      hcalIsoDeposit_Label = cms.untracked.InputTag("muIsoDepositCalByAssociatorTowers","hcal"),
                                      tkIsoDeposit_Label = cms.untracked.InputTag("muIsoDepositTk"),
                                      hoIsoDeposit_Label = cms.untracked.InputTag("muIsoDepositCalByAssociatorTowers","ho"),
                                      vertexLabel = cms.untracked.InputTag("offlinePrimaryVertices"),
                                      directory = cms.string("Muons_miniAOD/Isolation/standalone"),
                                     vtxBin = cms.int32(30),
                                     vtxMax = cms.double(149.5),
                                     vtxMin = cms.double(0.5)
                                      )

MuIsoDQM_glb_miniAOD = DQMEDAnalyzer('MuonIsolationDQM',
                                      Global_Muon_Label = cms.untracked.InputTag("slimmedMuons"),
                                      requireTRKMuon = cms.untracked.bool(False),
                                      requireSTAMuon = cms.untracked.bool(False),
                                      requireGLBMuon = cms.untracked.bool(True),
                                      ecalIsoDeposit_Label = cms.untracked.InputTag("muIsoDepositCalByAssociatorTowers","ecal"),
                                      #    rootfilename = cms.untracked.string('ttbar-DQMidation.root'),
                                      hcalIsoDeposit_Label = cms.untracked.InputTag("muIsoDepositCalByAssociatorTowers","hcal"),
                                      tkIsoDeposit_Label = cms.untracked.InputTag("muIsoDepositTk"),
                                      hoIsoDeposit_Label = cms.untracked.InputTag("muIsoDepositCalByAssociatorTowers","ho"),
                                      vertexLabel = cms.untracked.InputTag("offlinePrimaryVertices"),
                                      directory = cms.string("Muons_miniAOD/Isolation/global"),
                                     vtxBin = cms.int32(30),
                                     vtxMax = cms.double(149.5),
                                     vtxMin = cms.double(0.5),
                                      )
muIsoDQM_seq_miniAOD = cms.Sequence(MuIsoDQM_trk_miniAOD+MuIsoDQM_sta_miniAOD+MuIsoDQM_glb_miniAOD)

MuIsoDQM_glb_miniAOD_Phase2=MuIsoDQM_glb_miniAOD.clone(
    vtxBin=20,
    vtxMin=149.5,
    vtxMax=249.5                                                                                   
)
      
MuIsoDQM_trk_miniAOD_Phase2=MuIsoDQM_trk_miniAOD.clone(
    vtxBin=20,
    vtxMin=149.5,
    vtxMax=249.5                                                                                    
)

MuIsoDQM_sta_miniAOD_Phase2=MuIsoDQM_sta_miniAOD.clone(
    vtxBin=20,
    vtxMin=149.5,
    vtxMax=249.5                                                                                    
)

muIsoDQM_seq_miniAOD_Phase2 = cms.Sequence(MuIsoDQM_trk_miniAOD_Phase2+MuIsoDQM_sta_miniAOD_Phase2+MuIsoDQM_glb_miniAOD_Phase2)   

from Configuration.Eras.Modifier_phase2_muon_cff import phase2_muon                                                                         
phase2_muon.toReplaceWith(muIsoDQM_seq, muIsoDQM_seq_Phase2)
phase2_muon.toReplaceWith(muIsoDQM_seq_miniAOD, muIsoDQM_seq_miniAOD_Phase2)
