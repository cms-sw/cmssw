import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatAlgos.electronHLTProducer_cfi import *
from PhysicsTools.PatAlgos.muonHLTProducer_cfi import *
from PhysicsTools.PatAlgos.tauHLTProducer_cfi import *
from PhysicsTools.PatAlgos.photonHLTProducer_cfi import *
from PhysicsTools.PatAlgos.jetHLTProducer_cfi import *
from PhysicsTools.PatAlgos.metHLTProducer_cfi import *
from PhysicsTools.PatAlgos.electronHLTMatch_cfi import *
from PhysicsTools.PatAlgos.muonHLTMatch_cfi import *
from PhysicsTools.PatAlgos.tauHLTMatch_cfi import *
from PhysicsTools.PatAlgos.photonHLTMatch_cfi import *
from PhysicsTools.PatAlgos.jetHLTMatch_cfi import *
from PhysicsTools.PatAlgos.metHLTMatch_cfi import *
# include "PhysicsTools/PatAlgos/data/btagHLTProducer.cfi"
patHLTProducer_withoutTau = cms.Sequence(electronHLTProducer*muonHLTProducer*photonHLTProducer*jetHLTProducer*metHLTProducer)
patHLTProducer = cms.Sequence(patHLTProducer_withoutTau*tauHLTProducer)
# include "PhysicsTools/PatAlgos/data/btagHLTMatch.cfi"
patHLTMatch_withoutTau = cms.Sequence(electronHLTMatch*muonHLTMatch*photonHLTMatch*jetHLTMatch*metHLTMatch)
patHLTMatch = cms.Sequence(patHLTMatch_withoutTau*tauHLTMatch)
patHLT_withoutTau = cms.Sequence(patHLTProducer_withoutTau*patHLTMatch_withoutTau)
patHLT = cms.Sequence(patHLTProducer*patHLTMatch)

