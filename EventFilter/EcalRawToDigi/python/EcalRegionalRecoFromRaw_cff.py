import FWCore.ParameterSet.Config as cms

#common module
from EventFilter.EcalRawToDigi.EcalRawToRecHitFacility_cff import *
#FEDs + refGetter from region of interest.
from EventFilter.EcalRawToDigi.EcalRawToRecHitRoI_cff import *
# raw -> rechit for each path
from EventFilter.EcalRawToDigi.EcalRawToRecHitProducer_cff import *
# definition of sequences
ecalRegionalEgammaRecoSequence = cms.Sequence(EcalRawToRecHitFacility*ecalRegionalEgammaFEDs*ecalRegionalEgammaRecHit+cms.SequencePlaceholder("ecalPreshowerRecHit"))
ecalRegionalMuonsRecoSequence = cms.Sequence(EcalRawToRecHitFacility*ecalRegionalMuonsFEDs*ecalRegionalMuonsRecHit+cms.SequencePlaceholder("ecalPreshowerRecHit"))
ecalRegionalJetsRecoSequence = cms.Sequence(EcalRawToRecHitFacility*ecalRegionalJetsFEDs*ecalRegionalJetsRecHit+cms.SequencePlaceholder("ecalPreshowerRecHit"))
ecalRegionalTausRecoSequence = cms.Sequence(EcalRawToRecHitFacility*ecalRegionalTausFEDs*ecalRegionalTausRecHit+cms.SequencePlaceholder("ecalPreshowerRecHit"))
ecalAllRecoSequence = cms.Sequence(EcalRawToRecHitFacility*ecalRegionalRestFEDs*ecalRecHitAll+cms.SequencePlaceholder("ecalPreshowerRecHit"))

