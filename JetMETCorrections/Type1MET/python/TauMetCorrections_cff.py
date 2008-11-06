import FWCore.ParameterSet.Config as cms

# File: TauMetCorrections.cff
# Authors: C. N. Nguyen , A. Gurrola
# Date: 10.22.2007
#
# Met corrections for PFTaus

tauMetCorr = cms.EDProducer("TauMET",
   InputTausLabel   = cms.string('pfRecoTauProducer'),
   tauType   = cms.string('pfTau'),
   InputCaloJetsLabel = cms.string('iterativeCone5CaloJets'),
   jetPTthreshold = cms.double(20.0),
   jetEMfracLimit = cms.double(0.9),
   correctorLabel = cms.string('MCJetCorrectorIcone5'),
   InputMETLabel = cms.string('corMetType1Icone5'),
   metType = cms.string('recoCaloMET'),
   JetMatchDeltaR = cms.double(0.2),
   TauMinEt = cms.double(15.0),
   TauEtaMax = cms.double(2.5),
   UseSeedTrack = cms.bool(True),
   seedTrackPt = cms.double(6.0),
   UseTrackIsolation = cms.bool(True),
   trackIsolationMinPt = cms.double(1.0),
   UseECALIsolation = cms.bool(True),
   gammaIsolationMinPt = cms.double(1.5),
   UseProngStructure = cms.bool(False)
)

MetTauCorrections = cms.Sequence( tauMetCorr )

