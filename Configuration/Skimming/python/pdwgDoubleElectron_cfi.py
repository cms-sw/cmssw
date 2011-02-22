import FWCore.ParameterSet.Config as cms
import Configuration.Skimming.pdwgLeptonRecoSkim_cfi
DoubleElectron = Configuration.Skimming.pdwgLeptonRecoSkim_cfi.PdwgLeptonRecoSkim.clone()
DoubleElectron.UseElectronSelection = cms.bool(True)
DoubleElectron.electronPtMin = cms.double(10)
DoubleElectron.electronN = cms.int32(2)
DoubleElectron.filterName = cms.string("DoubleElectron")

