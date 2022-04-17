import FWCore.ParameterSet.Config as cms

from PhysicsTools.NanoAOD.nano_eras_cff import *
from  PhysicsTools.NanoAOD.common_cff import *


from RecoJets.JetProducers.ak4PFJets_cfi import ak4PFJets

chsForSATkJets = cms.EDFilter("CandPtrSelector", src = cms.InputTag("packedPFCandidates"), cut = cms.string('charge()!=0 && pvAssociationQuality()>=5 && vertexRef().key()==0'))
softActivityJets = ak4PFJets.clone(src = 'chsForSATkJets', doAreaFastjet = False, jetPtMin=1)
softActivityJets10 = cms.EDFilter("CandPtrSelector", src = cms.InputTag("softActivityJets"), cut = cms.string('pt>10'))
softActivityJets5 = cms.EDFilter("CandPtrSelector", src = cms.InputTag("softActivityJets"), cut = cms.string('pt>5'))
softActivityJets2 = cms.EDFilter("CandPtrSelector", src = cms.InputTag("softActivityJets"), cut = cms.string('pt>2'))


##### Soft Activity tables
saJetTable = cms.EDProducer("SimpleCandidateFlatTableProducer",
    src = cms.InputTag("softActivityJets"),
    cut = cms.string(""),
    maxLen = cms.uint32(6),
    name = cms.string("SoftActivityJet"),
    doc  = cms.string("jets clustered from charged candidates compatible with primary vertex (" + chsForSATkJets.cut.value()+")"),
    singleton = cms.bool(False), # the number of entries is variable
    extension = cms.bool(False), # this is the main table for the jets
    variables = cms.PSet(P3Vars,
  )
)

saJetTable.variables.pt.precision=10
saJetTable.variables.eta.precision=8
saJetTable.variables.phi.precision=8

saTable = cms.EDProducer("GlobalVariablesTableProducer",
    variables = cms.PSet(
        SoftActivityJetHT = ExtVar( cms.InputTag("softActivityJets"), "candidatescalarsum", doc = "scalar sum of soft activity jet pt, pt>1" ),
        SoftActivityJetHT10 = ExtVar( cms.InputTag("softActivityJets10"), "candidatescalarsum", doc = "scalar sum of soft activity jet pt , pt >10"  ),
        SoftActivityJetHT5 = ExtVar( cms.InputTag("softActivityJets5"), "candidatescalarsum", doc = "scalar sum of soft activity jet pt, pt>5"  ),
        SoftActivityJetHT2 = ExtVar( cms.InputTag("softActivityJets2"), "candidatescalarsum", doc = "scalar sum of soft activity jet pt, pt >2"  ),
        SoftActivityJetNjets10 = ExtVar( cms.InputTag("softActivityJets10"), "candidatesize", doc = "number of soft activity jet pt, pt >2"  ),
        SoftActivityJetNjets5 = ExtVar( cms.InputTag("softActivityJets5"), "candidatesize", doc = "number of soft activity jet pt, pt >5"  ),
        SoftActivityJetNjets2 = ExtVar( cms.InputTag("softActivityJets2"), "candidatesize", doc = "number of soft activity jet pt, pt >10"  ),

    )
)

softActivityTask = cms.Task(chsForSATkJets,softActivityJets,softActivityJets2,softActivityJets5,softActivityJets10)
softActivityTablesTask = cms.Task(saJetTable,saTable)
