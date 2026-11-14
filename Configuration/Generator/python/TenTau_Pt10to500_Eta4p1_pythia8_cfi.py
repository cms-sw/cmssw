import FWCore.ParameterSet.Config as cms

from Configuration.Generator.TenTau_Pt10to500_Eta3p1_pythia8_cfi import generator
generator.PGunParameters.MinEta = -4.1
generator.PGunParameters.MaxEta = 4.1
generator.psethack = cms.string("Ten tau leptons w/ uniform 10 < pT < 500 GeV, |eta| < 4.1")
