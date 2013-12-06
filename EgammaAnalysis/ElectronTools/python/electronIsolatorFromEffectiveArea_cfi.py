import FWCore.ParameterSet.Config as cms

elPFIsoValueEA03 = cms.EDFilter( "ElectronIsolatorFromEffectiveArea",
                                 gsfElectrons = cms.InputTag('gsfElectrons'),
                                 pfElectrons = cms.InputTag('pfSelectedElectrons'),
                                 rhoIso = cms.InputTag("kt6PFJets:rho"),
                                 EffectiveAreaType = cms.string("kEleGammaAndNeutralHadronIso03"),
                                 EffectiveAreaTarget = cms.string("kEleEAData2012")
                                 )
