import FWCore.ParameterSet.Config as cms
import Configuration.Skimming.pdwgLeptonRecoSkim_cfi
MuonPFElectron = Configuration.Skimming.pdwgLeptonRecoSkim_cfi.PdwgLeptonRecoSkim.clone()
MuonPFElectron.UsePfElectronSelection = cms.bool(True)
MuonPFElectron.UseMuonSelection = cms.bool(True)
MuonPFElectron.pfElectronPtMin = cms.double(10)
MuonPFElectron.pfElectronN = cms.int32(1)
MuonPFElectron.globalMuonPtMin = cms.double(5)
MuonPFElectron.trackerMuonPtMin = cms.double(10)
MuonPFElectron.muonN = cms.int32(1)
MuonPFElectron.filterName = cms.string("MuonPFElectron")

