import FWCore.ParameterSet.Config as cms
from RecoTauTag.RecoTau.PFRecoTauQualityCuts_cfi import PFTauQualityCuts
PFTauPrimaryVertexProducer = cms.EDProducer("PFTauPrimaryVertexProducer",
                                            PFTauTag =  cms.InputTag("hpsPFTauProducer"),
                                            ElectronTag = cms.InputTag("MyElectrons"),
                                            MuonTag = cms.InputTag("MyMuons"),
                                            PVTag = cms.InputTag("offlinePrimaryVertices"),
                                            beamSpot = cms.InputTag("offlineBeamSpot"),
                                            #Algorithm: 0 - use tau-jet vertex, 1 - use vertex[0]
                                            Algorithm = cms.int32(0),
                                            qualityCuts = PFTauQualityCuts,
                                            useBeamSpot = cms.bool(True),
                                            RemoveMuonTracks = cms.bool(False),
                                            RemoveElectronTracks = cms.bool(False),
                                            useSelectedTaus = cms.bool(False),
                                            discriminators = cms.VPSet(cms.PSet(discriminator = cms.InputTag('hpsPFTauDiscriminationByDecayModeFinding'),
                                            selectionCut = cms.double(0.5))),
                                            cut = cms.string("pt > 18.0 & abs(eta)<2.3")
                                            )

