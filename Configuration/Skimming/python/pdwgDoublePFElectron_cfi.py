import FWCore.ParameterSet.Config as cms
import Configuration.Skimming.pdwgLeptonRecoSkim_cfi
DoublePFElectron = Configuration.Skimming.pdwgLeptonRecoSkim_cfi.PdwgLeptonRecoSkim.clone()
DoublePFElectron.UsePfElectronSelection = cms.bool(True)
DoublePFElectron.pfElectronPtMin = cms.double(10)
DoublePFElectron.pfElectronN = cms.int32(2)
DoublePFElectron.filterName = cms.string("DoublePFElectron")

