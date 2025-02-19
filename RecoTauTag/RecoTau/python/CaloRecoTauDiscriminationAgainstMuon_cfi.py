import FWCore.ParameterSet.Config as cms

from RecoTauTag.RecoTau.TauDiscriminatorTools import requireLeadTrackCalo

caloRecoTauDiscriminationAgainstMuon = cms.EDProducer("CaloRecoTauDiscriminationAgainstMuon",

    # tau collection to discriminate
    CaloTauProducer = cms.InputTag('caloRecoTauProducer'),

    Prediscriminants = requireLeadTrackCalo,

    # algorithm parameters
    caloCompCoefficient = cms.double(0.5), ## user definde 2D Cut. Reject tau if calo * caloCompCoeff + segm * segmCompCoeff > cut 
    segmCompCoefficient = cms.double(0.5),
    muonCompCut = cms.double(0.0),
                                                    
    discriminatorOption = cms.string('noSegMatch'), ## available options are; noSegMatch, twoDCut, merePresence, combined

    muonSource = cms.InputTag("muons"),
    dRmatch = cms.double(0.5)                                             
)


