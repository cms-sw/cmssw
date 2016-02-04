import FWCore.ParameterSet.Config as cms
import Configuration.Skimming.pdwgLeptonRecoSkim_cfi
DoubleMu = Configuration.Skimming.pdwgLeptonRecoSkim_cfi.PdwgLeptonRecoSkim.clone()
DoubleMu.UseMuonSelection = cms.bool(True)
DoubleMu.muonN = cms.int32(2)
DoubleMu.globalMuonPtMin = cms.double(5)
DoubleMu.trackerMuonPtMin = cms.double(10)
DoubleMu.filterName = cms.string("DoubleMu")
