import FWCore.ParameterSet.Config as cms
import Configuration.Skimming.pdwgLeptonRecoSkim_cfi
SingleMu = Configuration.Skimming.pdwgLeptonRecoSkim_cfi.PdwgLeptonRecoSkim.clone()
SingleMu.UseMuonSelection = cms.bool(True)
SingleMu.globalMuonPtMin = cms.double(20)
SingleMu.trackerMuonPtMin = cms.double(20)
SingleMu.muonN = cms.int32(1)
SingleMu.filterName = cms.string("SingleMu")
