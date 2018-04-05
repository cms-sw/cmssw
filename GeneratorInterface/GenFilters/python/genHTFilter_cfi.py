#Configuration file fragment used for GenHTFilter module (GeneratorInterface/GenFilters/src/GenHTFilter.cc) initalisation
#genHTFilter_cfi GeneratorInterface/GenFilters/python/genHTFilter_cfi.py

import FWCore.ParameterSet.Config as cms

genHTFilter = cms.EDFilter("GenHTFilter",
   src = cms.InputTag("ak4GenJets"), #GenJet collection as input
   jetPtCut = cms.double(30.0), #GenJet pT cut for HT
   jetEtaCut = cms.double(4.5), #GenJet eta cut for HT
   genHTcut = cms.double(160.0) #genHT cut
)
