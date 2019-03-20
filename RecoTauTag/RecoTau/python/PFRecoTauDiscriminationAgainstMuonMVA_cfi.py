import FWCore.ParameterSet.Config as cms

from RecoTauTag.RecoTau.TauDiscriminatorTools import requireLeadTrack

pfRecoTauDiscriminationAgainstMuonMVA = cms.EDProducer("PFRecoTauDiscriminationAgainstMuonMVA",

    # tau collection to discriminate
    PFTauProducer = cms.InputTag('pfTauProducer'),

    # Require leading pion ensures that:
    #  1) these is at least one track above threshold (0.5 GeV) in the signal cone
    #  2) a track OR a pi-zero in the signal cone has pT > 5 GeV
    Prediscriminants = requireLeadTrack,
    loadMVAfromDB = cms.bool(True),
    inputFileName = cms.FileInPath("RecoTauTag/RecoTau/data/emptyMVAinputFile"), # the filename for MVA if it is not loaded from DB
    mvaName = cms.string("againstMuonMVA"),
    returnMVA = cms.bool(True),
    mvaMin = cms.double(0.0),

    srcMuons = cms.InputTag('muons'),
    dRmuonMatch = cms.double(0.3),
    verbosity = cms.int32(0),
)
