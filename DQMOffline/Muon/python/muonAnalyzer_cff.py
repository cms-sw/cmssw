import FWCore.ParameterSet.Config as cms
from Configuration.StandardSequences.Eras import eras

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
eras.phase1Pixel.toReplaceWith(muonAnalyzer, muonAnalyzer.copyAndExclude([ # FIXME
    muonRecoOneHLT # Doesn't work because TriggerResults::HLT is missing (because HLT not yet being part of 2017 workflow)
]))
eras.phase2_muon.toReplaceWith(muonAnalyzer, muonAnalyzer.copyAndExclude([ # FIXME
    muonEnergyDepositAnalyzer
]))

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
