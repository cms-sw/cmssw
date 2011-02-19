import FWCore.ParameterSet.Config as cms
import Configuration.Skimming.pdwgLeptonRecoSkim_cfi
MuonElectron = Configuration.Skimming.pdwgLeptonRecoSkim_cfi.PdwgLeptonRecoSkim.clone()
MuonElectron.UseElectronSelection = cms.bool(True)
MuonElectron.UseMuonSelection = cms.bool(True)
MuonElectron.electronPtMin = cms.double(10)
MuonElectron.electronN = cms.int32(1)
MuonElectron.globalMuonPtMin = cms.double(5)
MuonElectron.trackerMuonPtMin = cms.double(10)
MuonElectron.muonN = cms.int32(1)
MuonElectron.filterName = cms.string("MuonElectron")

