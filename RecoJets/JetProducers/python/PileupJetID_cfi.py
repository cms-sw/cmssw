import FWCore.ParameterSet.Config as cms

from RecoJets.JetProducers.PileupJetIDParams_cfi import * 

_stdalgos_4x = cms.VPSet(full,   cutbased,PhilV1)
_stdalgos_5x = cms.VPSet(full_5x,cutbased,PhilV1)

_chsalgos_4x = cms.VPSet(full,   cutbased)
_chsalgos_5x = cms.VPSet(full_5x_chs,cutbased)
_chsalgos = _chsalgos_5x

import os
try:
    cmssw_version = os.environ["CMSSW_VERSION"].replace("CMSSW_","")
except:
    cmssw_version = "5_X"

if cmssw_version.startswith("4"):
    _stdalgos    = _stdalgos_4x
    _chsalgos    = _chsalgos_4x
else:
    _stdalgos    = _stdalgos_5x
    _chsalgos    = _chsalgos_5x

pileupJetIdProducer = cms.EDProducer('PileupJetIdProducer',
                         produceJetIds = cms.bool(True),
                         jetids = cms.InputTag(""),
                         runMvas = cms.bool(True),
                         jets = cms.InputTag("selectedPatJetsPFlow"),
                         vertexes = cms.InputTag("offlinePrimaryVertices"),
                         algos = cms.VPSet(_stdalgos),
                                     
                         rho     = cms.InputTag("kt6PFJets","rho"),
                         jec     = cms.string("AK5PF"),
                         applyJec = cms.bool(False),
                         inputIsCorrected = cms.bool(True),                                     
                         residualsFromTxt = cms.bool(False),
                         residualsTxt     = cms.FileInPath("RecoJets/JetProducers/data/dummy.txt"),
)

pileupJetIdProducerChs = cms.EDProducer('PileupJetIdProducer',
                         produceJetIds = cms.bool(True),
                         jetids = cms.InputTag(""),
                         runMvas = cms.bool(True),
                         jets = cms.InputTag("selectedPatJetsPFlow"),
                         vertexes = cms.InputTag("offlinePrimaryVertices"),
                         algos = cms.VPSet(_chsalgos),
                                        
                         rho     = cms.InputTag("kt6PFJets","rho"),
                         jec     = cms.string("AK5PFchs"),
                         applyJec = cms.bool(False),
                         inputIsCorrected = cms.bool(True),
                         residualsFromTxt = cms.bool(False),
                         residualsTxt     = cms.FileInPath("RecoJets/JetProducers/data/dummy.txt"),

)

