import FWCore.ParameterSet.Config as cms

from RecoJets.JetProducers.ak4PFJetsPuppi_cfi import ak4PFJetsPuppi

ak8PFJetsPuppi = ak4PFJetsPuppi.clone(
	rParam = 0.8    
	)

