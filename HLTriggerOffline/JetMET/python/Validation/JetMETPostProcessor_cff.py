import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

JetMETPostVal = DQMEDHarvester("JetMETDQMPostProcessor",
     subDir = cms.untracked.string("HLT/HLTJETMET"),
     PatternJetTrg = cms.untracked.string("Jet([0-9])+"),
     PatternMetTrg = cms.untracked.string("M([E,H])+T([0-9])+")
       )
