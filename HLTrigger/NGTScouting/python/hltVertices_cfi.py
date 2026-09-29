import FWCore.ParameterSet.Config as cms
from PhysicsTools.NanoAOD.common_cff import *

hltVertexTable = cms.EDProducer("HLTVertexTableProducer",
                                skipNonExistingSrc = cms.bool(True),
                                usePF = cms.bool(True),
                                doSVs = cms.bool(True),
                                pvName = cms.string("hltPrimaryVertex"),
                                pvSrc = cms.InputTag("hltOfflinePrimaryVertices"),
                                pfSrc = cms.InputTag("hltParticleFlowTmp"),
                                goodPvCut = cms.string("!isFake && ndof >= 4.0 && abs(z) <= 24.0 && abs(position.Rho) <= 2.0"),
                                svName = cms.string("hltSecondaryVertex"),
                                svSrc = cms.InputTag("hltDeepInclusiveMergedVerticesPF"),
                                svDoc  = cms.string("secondary vertices from IVF algorithm"),
                                dlenMin = cms.double(0),
                                dlenSigMin = cms.double(3),
                                goodSvCut = cms.string(""))

hltPixelVertexTable = cms.EDProducer("HLTVertexTableProducer",
                                     skipNonExistingSrc = cms.bool(True),
                                     usePF = cms.bool(False), # use directly the tracks from PV fit
                                     doSVs = cms.bool(False), # no SVs built from pixel tracks
                                     pvName = cms.string("hltPixelVertex"),
                                     pvSrc = cms.InputTag("hltPhase2PixelVertices"),
                                     pfSrc = cms.InputTag(""),
                                     goodPvCut = cms.string(""))
