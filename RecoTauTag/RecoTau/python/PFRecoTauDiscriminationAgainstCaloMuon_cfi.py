import FWCore.ParameterSet.Config as cms

from RecoTauTag.RecoTau.TauDiscriminatorTools import requireLeadTrack

pfRecoTauDiscriminationAgainstCaloMuon = cms.EDProducer("PFRecoTauDiscriminationAgainstCaloMuon",
    
    # tau collection to discriminate
    PFTauProducer = cms.InputTag('pfRecoTauProducer'),

    # Require leading pion ensures that:
    #  1) these is at least one track above threshold (0.5 GeV) in the signal cone
    #  2) a track OR a pi-zero in the signal cone has pT > 5 GeV
    Prediscriminants = requireLeadTrack,

    # algorithm parameters
    srcEcalRecHitsBarrel = cms.InputTag('ecalRecHit', 'EcalRecHitsEB'),
    srcEcalRecHitsEndcap = cms.InputTag('ecalRecHit', 'EcalRecHitsEE'),
    srcHcalRecHits = cms.InputTag('hbhereco'),

    srcVertex = cms.InputTag('offlinePrimaryVerticesWithBS'),

    minLeadTrackPt = cms.double(15.), # GeV
    minLeadTrackPtFraction = cms.double(0.8), # leadTrackPt/sumPtSignalTracks

    dRecal = cms.double(15.), # cm (size of cylinder around lead. track in which ECAL energy deposits are summed)
    dRhcal = cms.double(25.), # cm (size of cylinder around lead. track in which HCAL energy deposits are summed)

    maxEnEcal = cms.double(3.), # GeV
    maxEnHcal = cms.double(8.), # GeV                                                    

    maxEnToTrackRatio = cms.double(0.25)
)


