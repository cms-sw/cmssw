import FWCore.ParameterSet.Config as cms

from DQMOffline.Muon.muonRecoOneHLT_cfi import *
from DQMOffline.Muon.muonEfficiencyAnalyzer_cfi import *
from DQMOffline.Muon.diMuonHistograms_cfi import *
from DQMOffline.Muon.muonKinVsEtaAnalyzer_cfi import *
from DQMOffline.Muon.muonRecoAnalyzer_cfi import *
from DQMOffline.Muon.muonEnergyDepositAnalyzer_cfi import *
from DQMOffline.Muon.segmentTrackAnalyzer_cfi import *
from DQMOffline.Muon.muonSeedsAnalyzer_cfi import *
from DQMOffline.Muon.muonPFAnalyzer_cfi import *

muonAnalyzer = cms.Sequence(muonEnergyDepositAnalyzer*
                            muonSeedsAnalyzer*
                            muonRecoAnalyzer*
                            glbMuonSegmentAnalyzer*
                            staMuonSegmentAnalyzer*
                            muonKinVsEtaAnalyzer*
                            diMuonHistos*
                            LooseMuonEfficiencyAnalyzer*
                            MediumMuonEfficiencyAnalyzer*
                            TightMuonEfficiencyAnalyzer*
                            muonPFsequence*
                            muonRecoOneHLT)
from Configuration.Eras.Modifier_phase1Pixel_cff import phase1Pixel
phase1Pixel.toReplaceWith(muonAnalyzer, muonAnalyzer.copyAndExclude([ # FIXME
    muonRecoOneHLT # Doesn't work because TriggerResults::HLT is missing (because HLT not yet being part of 2017 workflow)
]))
from Configuration.Eras.Modifier_phase2_muon_cff import phase2_muon
phase2_muon.toReplaceWith(muonAnalyzer, muonAnalyzer.copyAndExclude([ # FIXME
    muonEnergyDepositAnalyzer
]))

muonAnalyzer_miniAOD = cms.Sequence(muonRecoAnalyzer_miniAOD* 
                                    muonKinVsEtaAnalyzer_miniAOD*
                                    diMuonHistos_miniAOD*
                                    LooseMuonEfficiencyAnalyzer_miniAOD*
                                    MediumMuonEfficiencyAnalyzer_miniAOD*
                                    TightMuonEfficiencyAnalyzer_miniAOD)

muonAnalyzer_noHLT = cms.Sequence(muonEnergyDepositAnalyzer*
                                  muonSeedsAnalyzer*
                                  muonRecoAnalyzer*
                                  glbMuonSegmentAnalyzer*
                                  staMuonSegmentAnalyzer*
                                  muonKinVsEtaAnalyzer*
                                  diMuonHistos*
                                  LooseMuonEfficiencyAnalyzer*
                                  MediumMuonEfficiencyAnalyzer*
                                  TightMuonEfficiencyAnalyzer*                                
                                  muonPFsequence)

from Configuration.StandardSequences.Eras import eras
for e in [eras.pA_2016]:
  e.toModify(diMuonHistos, etaBin = cms.int32(175))
  e.toModify(diMuonHistos, etaBBin = cms.int32(175))
  e.toModify(diMuonHistos, etaEBin = cms.int32(175))
  e.toModify(diMuonHistos, etaBinLM = cms.int32(30))
  e.toModify(diMuonHistos, etaBBinLM = cms.int32(30))
  e.toModify(diMuonHistos, etaEBinLM = cms.int32(30))
  e.toModify(diMuonHistos, LowMassMin = cms.double(2.0))
  e.toModify(diMuonHistos, LowMassMax = cms.double(14.0))
  e.toModify(diMuonHistos, HighMassMin = cms.double(55.0))
  e.toModify(diMuonHistos, HighMassMax = cms.double(125.0))
