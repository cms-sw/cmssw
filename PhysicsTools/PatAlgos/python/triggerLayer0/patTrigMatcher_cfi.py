import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatAlgos.triggerLayer0.patTrigProducer_cfi import *

# Examples for configurations of the trigger match for various physics objects
# (cuts are NOT tuned, using old values from TQAF MC match, january 2008)


## matches to Egamma triggers

# matches to CandHLT1ElectronStartup

electronTrigMatchCandHLT1ElectronStartup = cms.EDProducer("PATTrigMatcher",
    src               = cms.InputTag("allLayer0Electrons"),
    matched           = cms.InputTag("patCandHLT1ElectronStartup"),
    maxDeltaR           = cms.double(0.2),
    maxDPtRel           = cms.double(1.0),
    resolveAmbiguities    = cms.bool(True),
    resolveByMatchQuality = cms.bool(False)
)

patTrigMatchCandHLT1ElectronStartup = cms.Sequence(
    patCandHLT1ElectronStartup *
    electronTrigMatchCandHLT1ElectronStartup
)

# matches to HLT1Photon

photonTrigMatchHLT1Photon = cms.EDProducer("PATTrigMatcher",
    src               = cms.InputTag("allLayer0Photons"),
    matched           = cms.InputTag("patHLT1Photon"),
    maxDeltaR           = cms.double(0.2),
    maxDPtRel           = cms.double(1.0),
    resolveAmbiguities    = cms.bool(True),
    resolveByMatchQuality = cms.bool(False)
)

patTrigMatchHLT1Photon = cms.Sequence(
    patHLT1Photon *
    photonTrigMatchHLT1Photon
)

# matches to HLT1PhotonRelaxed

photonTrigMatchHLT1PhotonRelaxed = cms.EDProducer("PATTrigMatcher",
     src               = cms.InputTag("allLayer0Photons"),
     matched           = cms.InputTag("patHLT1PhotonRelaxed"),
     maxDeltaR           = cms.double(0.2),
     maxDPtRel           = cms.double(1.0),
     resolveAmbiguities    = cms.bool(True),
     resolveByMatchQuality = cms.bool(False)
)

patTrigMatchHLT1PhotonRelaxed = cms.Sequence(
    patHLT1PhotonRelaxed *
    photonTrigMatchHLT1PhotonRelaxed
)

# matches to HLT2Photon

photonTrigMatchHLT2Photon = cms.EDProducer("PATTrigMatcher",
     src               = cms.InputTag("allLayer0Photons"),
     matched           = cms.InputTag("patHLT2Photon"),
     maxDeltaR           = cms.double(0.2),
     maxDPtRel           = cms.double(1.0),
     resolveAmbiguities    = cms.bool(True),
     resolveByMatchQuality = cms.bool(False)
)

patTrigMatchHLT2Photon = cms.Sequence(
    patHLT2Photon *
    photonTrigMatchHLT2Photon
)

# matches to HLT2PhotonRelaxed

photonTrigMatchHLT2PhotonRelaxed = cms.EDProducer("PATTrigMatcher",
     src               = cms.InputTag("allLayer0Photons"),
     matched           = cms.InputTag("patHLT2PhotonRelaxed"),
     maxDeltaR           = cms.double(0.2),
     maxDPtRel           = cms.double(1.0),
     resolveAmbiguities    = cms.bool(True),
     resolveByMatchQuality = cms.bool(False)
)

patTrigMatchHLT2PhotonRelaxed = cms.Sequence(
    patHLT2PhotonRelaxed *
    photonTrigMatchHLT2PhotonRelaxed
)

# matches to HLT1Electron

electronTrigMatchHLT1Electron = cms.EDProducer("PATTrigMatcher",
   src               = cms.InputTag("allLayer0Electrons"),
   matched           = cms.InputTag("patHLT1Electron"),
   maxDeltaR           = cms.double(0.5),
   maxDPtRel           = cms.double(0.5),
   resolveAmbiguities    = cms.bool(True),
   resolveByMatchQuality = cms.bool(False)
)

patTrigMatchHLT1Electron = cms.Sequence(
    patHLT1Electron *
    electronTrigMatchHLT1Electron
)

# matches to HLT1ElectronRelaxed
# including example of "wrong" match (jets which fired electron trigger)

electronTrigMatchHLT1ElectronRelaxed = cms.EDProducer("PATTrigMatcher",
   src               = cms.InputTag("allLayer0Electrons"),
   matched           = cms.InputTag("patHLT1ElectronRelaxed"),
   maxDeltaR           = cms.double(0.5),
   maxDPtRel           = cms.double(0.5),
   resolveAmbiguities    = cms.bool(True),
   resolveByMatchQuality = cms.bool(False)
)

jetTrigMatchHLT1ElectronRelaxed = cms.EDProducer("PATTrigMatcher",
   src               = cms.InputTag("allLayer0Jets"),
   matched           = cms.InputTag("patHLT1ElectronRelaxed"),
   maxDeltaR           = cms.double(0.5),
   maxDPtRel           = cms.double(0.5),
   resolveAmbiguities    = cms.bool(True),
   resolveByMatchQuality = cms.bool(False)
)

patTrigMatchHLT1ElectronRelaxed = cms.Sequence(
    patHLT1ElectronRelaxed *
    ( 
        electronTrigMatchHLT1ElectronRelaxed +
        jetTrigMatchHLT1ElectronRelaxed
    )
) 

# matches to HLT2Electron

electronTrigMatchHLT2Electron = cms.EDProducer("PATTrigMatcher",
   src               = cms.InputTag("allLayer0Electrons"),
   matched           = cms.InputTag("patHLT2Electron"),
   maxDeltaR           = cms.double(0.5),
   maxDPtRel           = cms.double(0.5),
   resolveAmbiguities    = cms.bool(True),
   resolveByMatchQuality = cms.bool(False)
)

patTrigMatchHLT2Electron = cms.Sequence(
    patHLT2Electron *
    electronTrigMatchHLT2Electron
)

# matches to HLT2ElectronRelaxed

electronTrigMatchHLT2ElectronRelaxed = cms.EDProducer("PATTrigMatcher",
   src               = cms.InputTag("allLayer0Electrons"),
   matched           = cms.InputTag("patHLT2ElectronRelaxed"),
   maxDeltaR           = cms.double(0.5),
   maxDPtRel           = cms.double(0.5),
   resolveAmbiguities    = cms.bool(True),
   resolveByMatchQuality = cms.bool(False)
)

patTrigMatchHLT2ElectronRelaxed = cms.Sequence(
    patHLT2ElectronRelaxed *
    electronTrigMatchHLT2ElectronRelaxed
)


## matches to Muon triggers

# matches to HLT1MuonIso

muonTrigMatchHLT1MuonIso = cms.EDProducer("PATTrigMatcher",
   src               = cms.InputTag("allLayer0Muons"),
   matched           = cms.InputTag("patHLT1MuonIso"),
   maxDeltaR           = cms.double(0.5),
   maxDPtRel           = cms.double(0.5),
   resolveAmbiguities    = cms.bool(True),
   resolveByMatchQuality = cms.bool(False)
)

patTrigMatchHLT1MuonIso = cms.Sequence(
    patHLT1MuonIso *
    muonTrigMatchHLT1MuonIso
)

# matches to HLT1MuonNonIso

muonTrigMatchHLT1MuonNonIso = cms.EDProducer("PATTrigMatcher",
   src               = cms.InputTag("allLayer0Muons"),
   matched           = cms.InputTag("patHLT1MuonNonIso"),
   maxDeltaR           = cms.double(0.5),
   maxDPtRel           = cms.double(0.5),
   resolveAmbiguities    = cms.bool(True),
   resolveByMatchQuality = cms.bool(False)
)

patTrigMatchHLT1MuonNonIso = cms.Sequence(
    patHLT1MuonNonIso *
    muonTrigMatchHLT1MuonNonIso
)

# matches to HLT2MuonNonIso

muonTrigMatchHLT2MuonNonIso = cms.EDProducer("PATTrigMatcher",
   src               = cms.InputTag("allLayer0Muons"),
   matched           = cms.InputTag("patHLT2MuonNonIso"),
   maxDeltaR           = cms.double(0.5),
   maxDPtRel           = cms.double(0.5),
   resolveAmbiguities    = cms.bool(True),
   resolveByMatchQuality = cms.bool(False)
)

patTrigMatchHLT2MuonNonIso = cms.Sequence(
    patHLT2MuonNonIso *
    muonTrigMatchHLT2MuonNonIso
)


## matches to BTau triggers

# matches to HLT1Tau

tauTrigMatchHLT1Tau = cms.EDProducer("PATTrigMatcher",
   src               = cms.InputTag("allLayer0Taus"),
   matched           = cms.InputTag("patHLT1Tau"),
   maxDeltaR           = cms.double(0.5),
   maxDPtRel           = cms.double(0.5),
   resolveAmbiguities    = cms.bool(True),
   resolveByMatchQuality = cms.bool(False)
)
 
patTrigMatchHLT1Tau = cms.Sequence(
    patHLT1Tau *
    tauTrigMatchHLT1Tau
)

# matches to HLT2TauPixel

tauTrigMatchHLT2TauPixel = cms.EDProducer("PATTrigMatcher",
   src               = cms.InputTag("allLayer0Taus"),
   matched           = cms.InputTag("patHLT2TauPixel"),
   maxDeltaR           = cms.double(0.5),
   maxDPtRel           = cms.double(0.5),
   resolveAmbiguities    = cms.bool(True),
   resolveByMatchQuality = cms.bool(False)
)
 
patTrigMatchHLT2TauPixel = cms.Sequence(
    patHLT2TauPixel *
    tauTrigMatchHLT2TauPixel
)


## matches to JetMET triggers

# matches to HLT2jet

jetTrigMatchHLT2jet = cms.EDProducer("PATTrigMatcher",
   src               = cms.InputTag("allLayer0Jets"),
   matched           = cms.InputTag("patHLT2jet"),
   maxDeltaR           = cms.double(0.4),
   maxDPtRel           = cms.double(3.0),
   resolveAmbiguities    = cms.bool(True),
   resolveByMatchQuality = cms.bool(False)
)

patTrigMatchHLT2jet = cms.Sequence(
    patHLT2jet *
    jetTrigMatchHLT2jet
)

# matches to HLT3jet

jetTrigMatchHLT3jet = cms.EDProducer("PATTrigMatcher",
   src               = cms.InputTag("allLayer0Jets"),
   matched           = cms.InputTag("patHLT3jet"),
   maxDeltaR           = cms.double(0.4),
   maxDPtRel           = cms.double(3.0),
   resolveAmbiguities    = cms.bool(True),
   resolveByMatchQuality = cms.bool(False)
)

patTrigMatchHLT3jet = cms.Sequence(
    patHLT3jet *
    jetTrigMatchHLT3jet
)

# matches to HLT4jet

jetTrigMatchHLT4jet = cms.EDProducer("PATTrigMatcher",
   src               = cms.InputTag("allLayer0Jets"),
   matched           = cms.InputTag("patHLT4jet"),
   maxDeltaR           = cms.double(0.4),
   maxDPtRel           = cms.double(3.0),
   resolveAmbiguities    = cms.bool(True),
   resolveByMatchQuality = cms.bool(False)
)

patTrigMatchHLT4jet = cms.Sequence(
    patHLT4jet *
    jetTrigMatchHLT4jet
)

# matches to HLT1MET65
# including example of "wrong" match (muons which fired MET trigger)

metTrigMatchHLT1MET65 = cms.EDProducer("PATTrigMatcher",
   src               = cms.InputTag("allLayer0METs"),
   matched           = cms.InputTag("patHLT1MET65"),
   maxDeltaR           = cms.double(0.4),
   maxDPtRel           = cms.double(3.0),
   resolveAmbiguities    = cms.bool(True),
   resolveByMatchQuality = cms.bool(False)
)

muonTrigMatchHLT1MET65 = cms.EDProducer("PATTrigMatcher",
   src               = cms.InputTag("allLayer0Muons"),
   matched           = cms.InputTag("patHLT1MET65"),
   maxDeltaR           = cms.double(0.4),
   maxDPtRel           = cms.double(3.0),
   resolveAmbiguities    = cms.bool(True),
   resolveByMatchQuality = cms.bool(False)
)

patTrigMatchHLT1MET65 = cms.Sequence(
    patHLT1MET65 *
    (
        metTrigMatchHLT1MET65 +
        muonTrigMatchHLT1MET65
    )
)


## matches to Xchannel triggers


## matches to Special triggers
