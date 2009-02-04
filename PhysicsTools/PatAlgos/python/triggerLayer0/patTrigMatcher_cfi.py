import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatAlgos.triggerLayer0.patTrigProducer_cfi import *
# Examples for configurations of the trigger match for various physics objects
# (cuts are NOT tuned, using old values from TQAF MC match, january 2008)


# matches to Egamma triggers
# matches to CandHLT1ElectronStartup
electronTrigMatchCandHLT1ElectronStartup = cms.EDFilter("PATTrigMatcher",
    src     = cms.InputTag("allLayer0Electrons"),
    matched = cms.InputTag("patCandHLT1ElectronStartup"),
    maxDPtRel = cms.double(1.0),
    maxDeltaR = cms.double(0.2),
    resolveAmbiguities    = cms.bool(True),
    resolveByMatchQuality = cms.bool(False),
)

# matches to HLT1Photon
photonTrigMatchHLT1Photon = cms.EDFilter("PATTrigMatcher",
    src     = cms.InputTag("allLayer0Photons"),
    matched = cms.InputTag("patHLT1Photon"),
    maxDPtRel = cms.double(1.0),
    maxDeltaR = cms.double(0.2),
    resolveAmbiguities    = cms.bool(True),
    resolveByMatchQuality = cms.bool(False),
)

# matches to HLT1PhotonRelaxed
photonTrigMatchHLT1PhotonRelaxed = cms.EDFilter("PATTrigMatcher",
    src     = cms.InputTag("allLayer0Photons"),
    matched = cms.InputTag("patHLT1PhotonRelaxed"),
    maxDPtRel = cms.double(1.0),
    maxDeltaR = cms.double(0.2),
    resolveAmbiguities    = cms.bool(True),
    resolveByMatchQuality = cms.bool(False),
)

# matches to HLT2Photon
photonTrigMatchHLT2Photon = cms.EDFilter("PATTrigMatcher",
    src     = cms.InputTag("allLayer0Photons"),
    matched = cms.InputTag("patHLT2Photon"),
    maxDPtRel = cms.double(1.0),
    maxDeltaR = cms.double(0.2),
    resolveAmbiguities    = cms.bool(True),
    resolveByMatchQuality = cms.bool(False),
)

# matches to HLT2PhotonRelaxed
photonTrigMatchHLT2PhotonRelaxed = cms.EDFilter("PATTrigMatcher",
    src     = cms.InputTag("allLayer0Photons"),
    matched = cms.InputTag("patHLT2PhotonRelaxed"),
    maxDPtRel = cms.double(1.0),
    maxDeltaR = cms.double(0.2),
    resolveAmbiguities    = cms.bool(True),
    resolveByMatchQuality = cms.bool(False),
)

# matches to HLT1Electron
electronTrigMatchHLT1Electron = cms.EDFilter("PATTrigMatcher",
    src     = cms.InputTag("allLayer0Electrons"),
    matched = cms.InputTag("patHLT1Electron"),
    maxDPtRel = cms.double(1.0),
    maxDeltaR = cms.double(0.2),
    resolveAmbiguities    = cms.bool(True),
    resolveByMatchQuality = cms.bool(False),
)

# matches to HLT1ElectronRelaxed
# including example of "wrong" match (jets which fired electron trigger),
electronTrigMatchHLT1ElectronRelaxed = cms.EDFilter("PATTrigMatcher",
    src     = cms.InputTag("allLayer0Electrons"),
    matched = cms.InputTag("patHLT1ElectronRelaxed"),
    maxDPtRel = cms.double(1.0),
    maxDeltaR = cms.double(0.2),
    resolveAmbiguities    = cms.bool(True),
    resolveByMatchQuality = cms.bool(False),
)

jetTrigMatchHLT1ElectronRelaxed = cms.EDFilter("PATTrigMatcher",
    src     = cms.InputTag("allLayer0Jets"),
    matched = cms.InputTag("patHLT1ElectronRelaxed"),
    maxDPtRel = cms.double(1.0),
    maxDeltaR = cms.double(0.2),
    resolveAmbiguities    = cms.bool(True),
    resolveByMatchQuality = cms.bool(False),
)

# matches to HLT2Electron
electronTrigMatchHLT2Electron = cms.EDFilter("PATTrigMatcher",
    src     = cms.InputTag("allLayer0Electrons"),
    matched = cms.InputTag("patHLT2Electron"),
    maxDPtRel = cms.double(1.0),
    maxDeltaR = cms.double(0.2),
    resolveAmbiguities    = cms.bool(True),
    resolveByMatchQuality = cms.bool(False),
)

# matches to HLT2ElectronRelaxed
electronTrigMatchHLT2ElectronRelaxed = cms.EDFilter("PATTrigMatcher",
    src     = cms.InputTag("allLayer0Electrons"),
    matched = cms.InputTag("patHLT2ElectronRelaxed"),
    maxDPtRel = cms.double(1.0),
    maxDeltaR = cms.double(0.2),
    resolveAmbiguities    = cms.bool(True),
    resolveByMatchQuality = cms.bool(False),
)

# matches to Muon triggers
# matches to HLT1MuonIso
muonTrigMatchHLT1MuonIso = cms.EDFilter("PATTrigMatcher",
    src     = cms.InputTag("allLayer0Muons"),
    matched = cms.InputTag("patHLT1MuonIso"),
    maxDPtRel = cms.double(1.0),
    maxDeltaR = cms.double(0.2),
    resolveAmbiguities    = cms.bool(True),
    resolveByMatchQuality = cms.bool(False),
)

# matches to HLT1MuonNonIso
muonTrigMatchHLT1MuonNonIso = cms.EDFilter("PATTrigMatcher",
    src     = cms.InputTag("allLayer0Muons"),
    matched = cms.InputTag("patHLT1MuonNonIso"),
    maxDPtRel = cms.double(1.0),
    maxDeltaR = cms.double(0.2),
    resolveAmbiguities    = cms.bool(True),
    resolveByMatchQuality = cms.bool(False),
)

# matches to HLT2MuonNonIso
muonTrigMatchHLT2MuonNonIso = cms.EDFilter("PATTrigMatcher",
    src     = cms.InputTag("allLayer0Muons"),
    matched = cms.InputTag("patHLT2MuonNonIso"),
    maxDPtRel = cms.double(1.0),
    maxDeltaR = cms.double(0.2),
    resolveAmbiguities    = cms.bool(True),
    resolveByMatchQuality = cms.bool(False),
)

# matches to BTau triggers
# matches to HLT1Tau
tauTrigMatchHLT1Tau = cms.EDFilter("PATTrigMatcher",
    src     = cms.InputTag("allLayer0Taus"),
    matched = cms.InputTag("patHLT1Tau"),
    maxDPtRel = cms.double(1.0),
    maxDeltaR = cms.double(0.2),
    resolveAmbiguities    = cms.bool(True),
    resolveByMatchQuality = cms.bool(False),
)

# matches to HLT2TauPixel
tauTrigMatchHLT2TauPixel = cms.EDFilter("PATTrigMatcher",
    src     = cms.InputTag("allLayer0Taus"),
    matched = cms.InputTag("patHLT2TauPixel"),
    maxDPtRel = cms.double(1.0),
    maxDeltaR = cms.double(0.2),
    resolveAmbiguities    = cms.bool(True),
    resolveByMatchQuality = cms.bool(False),
)

# matches to JetMET triggers
# matches to HLT2jet
jetTrigMatchHLT2jet = cms.EDFilter("PATTrigMatcher",
    src     = cms.InputTag("allLayer0Jets"),
    matched = cms.InputTag("patHLT2jet"),
    maxDPtRel = cms.double(1.0),
    maxDeltaR = cms.double(0.2),
    resolveAmbiguities    = cms.bool(True),
    resolveByMatchQuality = cms.bool(False),
)

# matches to HLT3jet
jetTrigMatchHLT3jet = cms.EDFilter("PATTrigMatcher",
    src     = cms.InputTag("allLayer0Jets"),
    matched = cms.InputTag("patHLT3jet"),
    maxDPtRel = cms.double(1.0),
    maxDeltaR = cms.double(0.2),
    resolveAmbiguities    = cms.bool(True),
    resolveByMatchQuality = cms.bool(False),
)

# matches to HLT4jet
jetTrigMatchHLT4jet = cms.EDFilter("PATTrigMatcher",
    src     = cms.InputTag("allLayer0Jets"),
    matched = cms.InputTag("patHLT4jet"),
    maxDPtRel = cms.double(1.0),
    maxDeltaR = cms.double(0.2),
    resolveAmbiguities    = cms.bool(True),
    resolveByMatchQuality = cms.bool(False),
)

# matches to HLT1MET65
# including example of "wrong" match (muons which fired MET trigger),
metTrigMatchHLT1MET65 = cms.EDFilter("PATTrigMatcher",
    src     = cms.InputTag("allLayer0METs"),
    matched = cms.InputTag("patHLT1MET65"),
    maxDPtRel = cms.double(1.0),
    maxDeltaR = cms.double(0.2),
    resolveAmbiguities    = cms.bool(True),
    resolveByMatchQuality = cms.bool(False),
)

muonTrigMatchHLT1MET65 = cms.EDFilter("PATTrigMatcher",
    src     = cms.InputTag("allLayer0Muons"),
    matched = cms.InputTag("patHLT1MET65"),
    maxDPtRel = cms.double(1.0),
    maxDeltaR = cms.double(0.2),
    resolveAmbiguities    = cms.bool(True),
    resolveByMatchQuality = cms.bool(False),
)

patTrigMatchCandHLT1ElectronStartup = cms.Sequence(patCandHLT1ElectronStartup * electronTrigMatchCandHLT1ElectronStartup)
patTrigMatchHLT1Photon = cms.Sequence(patHLT1Photon * photonTrigMatchHLT1Photon)
patTrigMatchHLT1PhotonRelaxed = cms.Sequence(patHLT1PhotonRelaxed * photonTrigMatchHLT1PhotonRelaxed)
patTrigMatchHLT2Photon = cms.Sequence(patHLT2Photon * photonTrigMatchHLT2Photon)
patTrigMatchHLT2PhotonRelaxed = cms.Sequence(patHLT2PhotonRelaxed * photonTrigMatchHLT2PhotonRelaxed)
patTrigMatchHLT1Electron = cms.Sequence(patHLT1Electron * electronTrigMatchHLT1Electron)
patTrigMatchHLT1ElectronRelaxed = cms.Sequence(patHLT1ElectronRelaxed * electronTrigMatchHLT1ElectronRelaxed + jetTrigMatchHLT1ElectronRelaxed)
patTrigMatchHLT2Electron = cms.Sequence(patHLT2Electron * electronTrigMatchHLT2Electron)
patTrigMatchHLT2ElectronRelaxed = cms.Sequence(patHLT2ElectronRelaxed * electronTrigMatchHLT2ElectronRelaxed)
patTrigMatchHLT1MuonIso = cms.Sequence(patHLT1MuonIso * muonTrigMatchHLT1MuonIso)
patTrigMatchHLT1MuonNonIso = cms.Sequence(patHLT1MuonNonIso * muonTrigMatchHLT1MuonNonIso)
patTrigMatchHLT2MuonNonIso = cms.Sequence(patHLT2MuonNonIso * muonTrigMatchHLT2MuonNonIso)
patTrigMatchHLT1Tau = cms.Sequence(patHLT1Tau * tauTrigMatchHLT1Tau)
patTrigMatchHLT2TauPixel = cms.Sequence(patHLT2TauPixel * tauTrigMatchHLT2TauPixel)
patTrigMatchHLT2jet = cms.Sequence(patHLT2jet * jetTrigMatchHLT2jet)
patTrigMatchHLT3jet = cms.Sequence(patHLT3jet * jetTrigMatchHLT3jet)
patTrigMatchHLT4jet   = cms.Sequence(patHLT4jet * jetTrigMatchHLT4jet)
patTrigMatchHLT1MET65 = cms.Sequence(patHLT1MET65 * metTrigMatchHLT1MET65 + muonTrigMatchHLT1MET65)


## patTuple ##

# matches to HLT_IsoMu11
muonTrigMatchHLTIsoMu11 = cms.EDFilter("PATTrigMatcher",
    src     = cms.InputTag("allLayer0Muons"),
    matched = cms.InputTag("patHLTIsoMu11"),
    maxDPtRel = cms.double(1.0),
    maxDeltaR = cms.double(0.2),
    resolveAmbiguities    = cms.bool(True),
    resolveByMatchQuality = cms.bool(False),
)

# matches to HLT_Mu11
muonTrigMatchHLTMu11 = cms.EDFilter("PATTrigMatcher",
    src     = cms.InputTag("allLayer0Muons"),
    matched = cms.InputTag("patHLTMu11"),
    maxDPtRel = cms.double(1.0),
    maxDeltaR = cms.double(0.2),
    resolveAmbiguities    = cms.bool(True),
    resolveByMatchQuality = cms.bool(False),
)

# matches to HLT_DoubleIsoMu3
muonTrigMatchHLTDoubleIsoMu3 = cms.EDFilter("PATTrigMatcher",
    src     = cms.InputTag("allLayer0Muons"),
    matched = cms.InputTag("patHLTDoubleIsoMu3"),
    maxDPtRel = cms.double(1.0),
    maxDeltaR = cms.double(0.2),
    resolveAmbiguities    = cms.bool(True),
    resolveByMatchQuality = cms.bool(False),
)

# matches to HLT_DoubleMu3
muonTrigMatchHLTDoubleMu3 = cms.EDFilter("PATTrigMatcher",
    src     = cms.InputTag("allLayer0Muons"),
    matched = cms.InputTag("patHLTDoubleMu3"),
    maxDPtRel = cms.double(1.0),
    maxDeltaR = cms.double(0.2),
    resolveAmbiguities    = cms.bool(True),
    resolveByMatchQuality = cms.bool(False),
)

# matches to HLT_IsoEle15_LW_L1I
electronTrigMatchHLTIsoEle15LWL1I = cms.EDFilter("PATTrigMatcher",
    src     = cms.InputTag("allLayer0Electrons"),
    matched = cms.InputTag("patHLTIsoEle15LWL1I"),
    maxDPtRel = cms.double(1.0),
    maxDeltaR = cms.double(0.2),
    resolveAmbiguities    = cms.bool(True),
    resolveByMatchQuality = cms.bool(False),
)

# matches to HLT_Ele15_LW_L1R
electronTrigMatchHLTEle15LWL1R = cms.EDFilter("PATTrigMatcher",
    src     = cms.InputTag("allLayer0Electrons"),
    matched = cms.InputTag("patHLTEle15LWL1R"),
    maxDPtRel = cms.double(1.0),
    maxDeltaR = cms.double(0.2),
    resolveAmbiguities    = cms.bool(True),
    resolveByMatchQuality = cms.bool(False),
)

# matches to HLT_DoubleIsoEle10_LW_L1I
electronTrigMatchHLTDoubleIsoEle10LWL1I = cms.EDFilter("PATTrigMatcher",
    src     = cms.InputTag("allLayer0Electrons"),
    matched = cms.InputTag("patHLTDoubleIsoEle10LWL1I"),
    maxDPtRel = cms.double(1.0),
    maxDeltaR = cms.double(0.2),
    resolveAmbiguities    = cms.bool(True),
    resolveByMatchQuality = cms.bool(False),
)

# matches to HLT_DoubleEle5_SW_L1R
electronTrigMatchHLTDoubleEle5SWL1R = cms.EDFilter("PATTrigMatcher",
    src     = cms.InputTag("allLayer0Electrons"),
    matched = cms.InputTag("patHLTDoubleEle5SWL1R"),
    maxDPtRel = cms.double(1.0),
    maxDeltaR = cms.double(0.2),
    resolveAmbiguities    = cms.bool(True),
    resolveByMatchQuality = cms.bool(False),
)

# matches to HLT_LooseIsoTau_MET30_L1MET
tauTrigMatchHLTLooseIsoTauMET30L1MET = cms.EDFilter("PATTrigMatcher",
    src     = cms.InputTag("allLayer0Taus"),
    matched = cms.InputTag("patHLTLooseIsoTauMET30L1MET"),
    maxDPtRel = cms.double(1.0),
    maxDeltaR = cms.double(0.2),
    resolveAmbiguities    = cms.bool(True),
    resolveByMatchQuality = cms.bool(False),
)

# matches to HLT_DoubleIsoTau_Trk3
tauTrigMatchHLTDoubleIsoTauTrk3 = cms.EDFilter("PATTrigMatcher",
    src     = cms.InputTag("allLayer0Taus"),
    matched = cms.InputTag("patHLTDoubleIsoTauTrk3"),
    maxDPtRel = cms.double(1.0),
    maxDeltaR = cms.double(0.2),
    resolveAmbiguities    = cms.bool(True),
    resolveByMatchQuality = cms.bool(False),
)

patTrigMatchHLT_IsoMu11 = cms.Sequence(patHLTIsoMu11 * muonTrigMatchHLTIsoMu11)
patTrigMatchHLT_Mu11 = cms.Sequence(patHLTMu11 * muonTrigMatchHLTMu11)
patTrigMatchHLT_DoubleIsoMu3 = cms.Sequence(patHLTDoubleIsoMu3 * muonTrigMatchHLTDoubleIsoMu3)
patTrigMatchHLT_DoubleMu3 = cms.Sequence(patHLTDoubleMu3 * muonTrigMatchHLTDoubleMu3)
patTrigMatchHLT_IsoEle15_LW_L1I = cms.Sequence(patHLTIsoEle15LWL1I * electronTrigMatchHLTIsoEle15LWL1I)
patTrigMatchHLT_Ele15_LW_L1R = cms.Sequence(patHLTEle15LWL1R * electronTrigMatchHLTEle15LWL1R)
patTrigMatchHLT_DoubleIsoEle10_LW_L1I = cms.Sequence(patHLTDoubleIsoEle10LWL1I * electronTrigMatchHLTDoubleIsoEle10LWL1I)
patTrigMatchHLT_DoubleEle5_SW_L1R = cms.Sequence(patHLTDoubleEle5SWL1R * electronTrigMatchHLTDoubleEle5SWL1R)
patTrigMatchHLT_LooseIsoTau_MET30_L1MET = cms.Sequence(patHLTLooseIsoTauMET30L1MET * tauTrigMatchHLTLooseIsoTauMET30L1MET)
patTrigMatchHLT_DoubleIsoTau_Trk3 = cms.Sequence(patHLTDoubleIsoTauTrk3 * tauTrigMatchHLTDoubleIsoTauTrk3)
