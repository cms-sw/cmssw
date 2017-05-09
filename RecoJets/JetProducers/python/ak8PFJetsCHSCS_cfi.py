import FWCore.ParameterSet.Config as cms

from RecoJets.JetProducers.ak4PFJets_cfi import ak4PFJets
from RecoJets.JetProducers.SubJetParameters_cfi import SubJetParameters

ak8PFJetsCHSCS = ak4PFJets.clone(
	src = cms.InputTag("pfNoPileUpJME"),
	rParam = cms.double( 0.8 ),
	useConstituentSubtraction = cms.bool(True),    
	csRParam = cms.double(0.4),
	useExplicitGhosts = cms.bool(True),
	doAreaFastjet = cms.bool(True),
	jetPtMin = cms.double(100.0)
	)
