import FWCore.ParameterSet.Config as cms

from RecoJets.JetProducers.ak4PFJets_cfi import ak4PFJets
from RecoJets.JetProducers.ak5PFJetsTrimmed_cfi import ak5PFJetsTrimmed
from RecoJets.JetProducers.ak5PFJetsFiltered_cfi import ak5PFJetsFiltered, ak5PFJetsMassDropFiltered
from RecoJets.JetProducers.ak5PFJetsPruned_cfi import ak5PFJetsPruned
from RecoJets.JetProducers.SubJetParameters_cfi import SubJetParameters

ak8PFJetsCS = ak4PFJets.clone(
	rParam = cms.double( 0.8 ),
	useConstituentSubtraction = cms.bool(True),    
	csRParam = cms.double(0.4),
	useExplicitGhosts = cms.bool(True),
	doAreaFastjet = cms.bool(True),
	jetPtMin = cms.double(100.0)
	)



ak8PFJetsCSConstituents = cms.EDProducer("PFJetConstituentSelector",
                                         src = cms.InputTag("ak8PFJetsCS"),
                                         cut = cms.string("pt > 100.0")
                                        )


ak8PFJetsCSPruned = ak5PFJetsPruned.clone(
	rParam = cms.double( 0.8 ),
	jetPtMin = cms.double(15.0),
	src = cms.InputTag("ak8PFJetsCSConstituents", "constituents")
	)

ak8PFJetsCSTrimmed = ak5PFJetsTrimmed.clone(
	rParam = cms.double( 0.8 ),
	jetPtMin = cms.double(15.0),
	src = cms.InputTag("ak8PFJetsCSConstituents", "constituents")
	)


ak8PFJetsCSFiltered = ak5PFJetsFiltered.clone(
	rParam = cms.double( 0.8 ),
	jetPtMin = cms.double(15.0),
	src = cms.InputTag("ak8PFJetsCSConstituents", "constituents")
	)

