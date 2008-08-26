import FWCore.ParameterSet.Config as cms

import JetMETCorrections.Type1MET.corMetMuons_cfi


goodMuonsforMETCorrection = cms.EDFilter("MuonSelector",
    src = cms.InputTag("muons"),
    cut = cms.string('isGlobalMuon=1 & pt > 10.0 & abs(eta)<2.5 & innerTrack.numberOfValidHits>5 & combinedMuon.qoverpError< 0.5')
)

corMetGlobalMuons = JetMETCorrections.Type1MET.corMetMuons_cfi.corMetMuons.clone()
##MetMuonCorrections = cms.Sequence(corMetGlobalMuons)
MetMuonCorrections = cms.Sequence(goodMuonsforMETCorrection*corMetGlobalMuons)
corMetGlobalMuons.TrackAssociatorParameters.useEcal = False
corMetGlobalMuons.TrackAssociatorParameters.useHcal = False
corMetGlobalMuons.TrackAssociatorParameters.useHO = False
corMetGlobalMuons.TrackAssociatorParameters.useCalo = True
corMetGlobalMuons.TrackAssociatorParameters.useMuon = False
corMetGlobalMuons.TrackAssociatorParameters.truthMatch = False
