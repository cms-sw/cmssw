import FWCore.ParameterSet.Config as cms

RPCChamberMasker = cms.EDProducer('RPCChamberMasker',
                                  digiTag = cms.InputTag('preRPCDigis'),
                                  descopeRE31 = cms.bool(False),
                                  descopeRE41 =cms.bool(False)
				  )

from Configuration.Eras.Modifier_phase2_muon_cff import phase2_muon

if phase2_muon.isChosen():
	RPCChamberMasker = cms.EDProducer('RPCChamberMasker',
					  digiTag = cms.InputTag('simMuonRPCDigis'),
					  descopeRE31 = cms.bool(False),
					  descopeRE41 =cms.bool(False)
					  )
