#Configuration file fragment used for ModelpMSSMFilter module (GeneratorInterface/GenFilters/plugins/ModelpMSSMFilter.cc) initalisation
#ModelpMSSMFilter_cfi GeneratorInterface/GenFilters/python/ModelpMSSMFilter_cfi.py

import FWCore.ParameterSet.Config as cms


ModelpMSSMFilter = cms.EDFilter("ModelpMSSMFilter",
	gpssrc = cms.InputTag("genParticles"),    # input genParticle collection
 	jetsrc = cms.InputTag("ak4GenJets"),  # input genJets collection
  genHTcut = cms.double(140.0),                # genHT cut
  jetEtaCut = cms.double(5.0),                 # genJet eta cut for HT
	jetPtCut = cms.double(30.0),                 # genJet pT cut for HT			   
	elEtaCut = cms.double(2.5),                  # gen electron eta cut for single electron trigger
	elPtCut = cms.double(15.0),                  # gen electron pT cut for single electron trigger
	gammaEtaCut = cms.double(2.5),               # gen photon eta cut for single photon trigger
	gammaPtCut = cms.double(70.0),               # gen photon pT cut for single photon trigger
	muEtaCut = cms.double(2.5),                  # gen muon eta cut for single muon trigger
	muPtCut = cms.double(15.0),                  # gen muon pT cut for single muon trigger
  tauEtaCut = cms.double(2.5),                 # gen tau eta cut for di-tau trigger
	tauPtCut = cms.double(30.0),                 # gen tau pT cut for di-tau trigger
	loosemuPtCut = cms.double(2.5),              # gen muon pT cut for soft object trigger
	looseelPtCut = cms.double(5.0),              # gen electron pT cut for soft object trigger
	loosegammaPtCut = cms.double(30.0),          # gen photon pT cut for soft object trigger
	veryloosegammaPtCut = cms.double(18.0)       # gen photon pT cut for di-photon trigger
  )
