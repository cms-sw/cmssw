import FWCore.ParameterSet.Config as cms
from PhysicsTools.NanoAOD.common_cff import *

hltVertexTable = cms.EDProducer("HLTVertexTableProducer",
                                skipNonExistingSrc = cms.bool(True),
                                pvSrc = cms.InputTag("hltOfflinePrimaryVertices"),
                                goodPvCut = cms.string("!isFake && ndof >= 4.0 && abs(z) <= 24.0 && abs(position.Rho) <= 2.0"), 
                                pfSrc = cms.InputTag("hltParticleFlowTmp"),
                                dlenMin = cms.double(0),
                                dlenSigMin = cms.double(3),
                                pvName = cms.string("hltPrimaryVertex"))

hltPixelVertexTable = cms.EDProducer("HLTVertexTableProducer",
                                     skipNonExistingSrc = cms.bool(True),
                                     pvSrc = cms.InputTag("hltPhase2PixelVertices"),
                                     goodPvCut = cms.string(""),
                                     usePF = cms.bool(False), # use directly the tracks from PV fit 
                                     pfSrc = cms.InputTag(""),
                                     dlenMin = cms.double(0),
                                     dlenSigMin = cms.double(3),
                                     pvName = cms.string("hltPixelVertex"))

hltSecondaryVertexTable = cms.EDProducer("SimpleSecondaryVertexFlatTableProducer",
                                         skipNonExistingSrc = cms.bool(False),
                                         src = cms.InputTag("hltDeepInclusiveMergedVerticesPF"),
                                         name = cms.string("hltSecondaryVertex"),
                                         extension = cms.bool(False),
                                         variables = cms.PSet(P4Vars,
                                                              x   = Var("position().x()", float, doc = "secondary vertex X position, in cm",precision=10),
                                                              y   = Var("position().y()", float, doc = "secondary vertex Y position, in cm",precision=10),
                                                              z   = Var("position().z()", float, doc = "secondary vertex Z position, in cm",precision=14),
                                                              ndof    = Var("vertexNdof()", float, doc = "number of degrees of freedom",precision=8),
                                                              chi2    = Var("vertexNormalizedChi2()", float, doc = "reduced chi2, i.e. chi/ndof",precision=8),
                                                              ntracks = Var("numberOfDaughters()", "uint8", doc = "number of tracks"),
                                                              ),
                                         )

hltSecondaryVertexTable.variables.pt.precision=10
hltSecondaryVertexTable.variables.phi.precision=12
