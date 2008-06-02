import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatAlgos.emL1Producer_cfi import *
from PhysicsTools.PatAlgos.muonL1Producer_cfi import *
# include "PhysicsTools/PatAlgos/data/jetL1Producer.cfi"
from PhysicsTools.PatAlgos.metL1Producer_cfi import *
from PhysicsTools.PatAlgos.emL1Match_cfi import *
from PhysicsTools.PatAlgos.muonL1Match_cfi import *
# include "PhysicsTools/PatAlgos/data/jetL1Match.cfi"
from PhysicsTools.PatAlgos.metL1Match_cfi import *
patL1Producer_withoutTau = cms.Sequence(emL1Producer*muonL1Producer*metL1Producer)
patL1Producer = cms.Sequence(patL1Producer_withoutTau)
patL1Match_withoutTau = cms.Sequence(emL1Match*muonL1Match*metL1Match)
patL1Match = cms.Sequence(patL1Match_withoutTau)
patL1_withoutTau = cms.Sequence(patL1Producer_withoutTau*patL1Match_withoutTau)
patL1 = cms.Sequence(patL1Producer*patL1Match)

