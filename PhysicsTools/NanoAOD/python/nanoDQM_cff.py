import FWCore.ParameterSet.Config as cms

from PhysicsTools.NanoAOD.nanoDQM_cfi import nanoDQM

nanoDQMMC = nanoDQM.clone()
nanoDQMMC.vplots.Electron.sels.Prompt = cms.string("mcMatchFlav == 1")
nanoDQMMC.vplots.Muon.sels.Prompt = cms.string("mcMatchFlav == 1")
nanoDQMMC.vplots.Photon.sels.Prompt = cms.string("mcMatchFlav == 1")
nanoDQMMC.vplots.Tau.sels.Prompt = cms.string("mcMatchFlav == 5")
nanoDQMMC.vplots.Jet.sels.Prompt = cms.string("genJetIdx != 1")
nanoDQMMC.vplots.Jet.sels.PromptB = cms.string("genJetIdx != 1 && hadronFlavour == 5")

nanoDQMSequence = cms.Sequence(nanoDQM)
nanoDQMSequenceMC = cms.Sequence(nanoDQMMC)
