import FWCore.ParameterSet.Config as cms

from RecoTauTag.RecoTau.TauDiscriminatorTools import requireLeadTrack

pfRecoTauDiscriminationAgainstMuon2 = cms.EDProducer("PFRecoTauDiscriminationAgainstMuon2",
    
    # tau collection to discriminate
    PFTauProducer = cms.InputTag('pfRecoTauProducer'),

    # Require leading pion ensures that:
    #  1) these is at least one track above threshold (0.5 GeV) in the signal cone
    #  2) a track OR a pi-zero in the signal cone has pT > 5 GeV
    Prediscriminants = requireLeadTrack,

    # algorithm parameters
    discriminatorOption = cms.string('loose'), # available options are: 'loose', 'medium', 'tight' and 'custom'
    HoPMin = cms.double(0.2),
    maxNumberOfMatches = cms.int32(0), # negative value would turn off this cut in case of 'custom' discriminator 
    doCaloMuonVeto = cms.bool(False),
    maxNumberOfHitsLast2Stations = cms.int32(0), # negative value would turn off this cut in case of 'custom' discriminator 

    # optional collection of muons to check for overlap with taus
    srcMuons = cms.InputTag('muons'),
    dRmuonMatch = cms.double(0.3),                                                    
    dRmuonMatchLimitedToJetArea = cms.bool(False),
    minPtMatchedMuon = cms.double(5.),

    # flags to mask/unmask DT, CSC and RPC chambers in individual muon stations.
    # Segments and hits that are present in that muon station are ignored in case the "mask" is set to 1.
    # Per default only the innermost CSC chamber is ignored, as it is affected by spurious hits in high pile-up events
    maskMatchesDT = cms.vint32(0,0,0,0),
    maskMatchesCSC = cms.vint32(1,0,0,0),
    maskMatchesRPC = cms.vint32(0,0,0,0),
    maskHitsDT = cms.vint32(0,0,0,0),
    maskHitsCSC = cms.vint32(0,0,0,0),
    maskHitsRPC = cms.vint32(0,0,0,0),

    verbosity = cms.int32(0)
)


