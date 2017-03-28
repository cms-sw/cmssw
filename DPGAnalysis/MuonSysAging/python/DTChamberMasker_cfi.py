
import FWCore.ParameterSet.Config as cms

DTChamberMasker = cms.EDProducer('DTChamberMasker',
                                 digiTag = cms.InputTag('simMuonDTDigis')
)
