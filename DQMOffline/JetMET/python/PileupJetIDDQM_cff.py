import FWCore.ParameterSet.Config as cms
from RecoJets.JetProducers.PileupJetIDParams_cfi import * 

_stdalgos_5x = cms.VPSet(full_5x_chs,cutbased)
#_chsalgos_5x = cms.VPSet(full_5x_chs,cutbased)

#_chsalgos = _chsalgos_5x

_stdalgos = _stdalgos_5x
#_chsalgos = _chsalgos_5x

pileupJetIdProducer = cms.EDProducer('PileupJetIdProducer',
                   produceJetIds = cms.bool(True),
                   jetids = cms.InputTag(""),
                   runMvas = cms.bool(True),
                   #jets = cms.InputTag("selectedPatJetsPFlow"),
                   jets = cms.InputTag("ak4PFJets"),
                   vertexes = cms.InputTag("offlinePrimaryVertices"),
                   algos = cms.VPSet(_stdalgos),
                   rho = cms.InputTag("fixedGridRhoFastjetAll"),
                   jec = cms.string("AK4PF"),
                   applyJec = cms.bool(False),
                   inputIsCorrected = cms.bool(True),
                   residualsFromTxt = cms.bool(False),
                   residualsTxt = cms.FileInPath("RecoJets/JetProducers/data/download.url"),
)
