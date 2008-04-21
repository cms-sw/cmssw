import FWCore.ParameterSet.Config as cms

from RecoJets.Configuration.RecoJets_cff import *
from RecoMET.Configuration.RecoMET_cff import *
from JetMETCorrections.Configuration.MCJetCorrectionsHLT_cff import *
from RecoMET.Configuration.RecoHTMET_cff import *
from HLTrigger.HLTfilters.hlt1CaloJetDefaults_cff import *
import copy
from RecoJets.JetProducers.iterativeCone5CaloJets_cff import *
iterativeCone5CaloJetsRegional = copy.deepcopy(iterativeCone5CaloJets)
from HLTrigger.HLTfilters.hlt1CaloJetRegionalDefaults_cff import *
MCJetCorJetIcone5Regional = cms.EDProducer("JetCorrectionProducer",
    src = cms.InputTag("iterativeCone5CaloJetsRegional"),
    correctors = cms.vstring('MCJetCorrectorIcone5'),
    alias = cms.untracked.string('corJetIcone5')
)

doHLTJetReco = cms.Sequence(iterativeCone5CaloJets+MCJetCorJetIcone5)
doHLTMETReco = cms.Sequence(met)
doHLTHTReco = cms.Sequence(htMet)
doRegionalHLTJetReco = cms.Sequence(iterativeCone5CaloJetsRegional+MCJetCorJetIcone5Regional)
htMet.src = 'MCJetCorJetIcone5'
hlt1CaloJetDefaults.inputTag = 'MCJetCorJetIcone5'
iterativeCone5CaloJetsRegional.src = 'caloTowersForJets'
hlt1CaloJetRegionalDefaults.inputTag = 'MCJetCorJetIcone5Regional'

