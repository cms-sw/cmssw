import FWCore.ParameterSet.Config as cms

#needed for Likelihood tagger
from RecoJets.JetProducers.kt4PFJets_cfi import *
kt6PFJetsIsoQG = kt4PFJets.clone( rParam = 0.6, doRhoFastjet = True )
kt6PFJetsIsoQG.Rho_EtaMax = cms.double(2.5)

QGTagger = cms.EDProducer('QGTagger',
  srcRhoIso     = cms.InputTag('kt6PFJetsIsoQG','rho'),
  jec		= cms.string(''),
  dataDir	= cms.string('CondFormats/JetMETObjects/data/'),
  useCHS	= cms.bool(True)
)

