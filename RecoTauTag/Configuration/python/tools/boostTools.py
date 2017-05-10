import FWCore.ParameterSet.Config as cms

def clonePFTau(process,postfix="Boost"):
   process.load("RecoTauTag.Configuration.RecoPFTauTag_cff")
   from PhysicsTools.PatAlgos.tools.helpers import cloneProcessingSnippet
   cloneProcessingSnippet(process, process.PFTau, postfix)

   from PhysicsTools.PatAlgos.tools.pfTools import adaptPFTaus

#   adaptPFTaus(process,'hpsPFTau',postfix)

   getattr(process,"recoTauAK5PFJets08Region"+postfix).src = cms.InputTag('boostedTauSeeds')
   getattr(process,"recoTauAK5PFJets08Region"+postfix).pfCandSrc = cms.InputTag('pfNoPileUpForBoostedTaus')
   getattr(process,"recoTauAK5PFJets08Region"+postfix).pfCandAssocMapSrc = cms.InputTag('boostedTauSeeds', 'pfCandAssocMapForIsolation')

   getattr(process,"ak5PFJetsLegacyHPSPiZeros"+postfix).jetSrc = cms.InputTag('boostedTauSeeds')

   getattr(process,"ak5PFJetsRecoTauChargedHadrons"+postfix).jetSrc = cms.InputTag('boostedTauSeeds')
   getattr(process,"ak5PFJetsRecoTauChargedHadrons"+postfix).builders[1].dRcone = cms.double(0.3)
   getattr(process,"ak5PFJetsRecoTauChargedHadrons"+postfix).builders[1].dRconeLimitedToJetArea = cms.bool(True)

   getattr(process,"combinatoricRecoTaus"+postfix).jetSrc = cms.InputTag('boostedTauSeeds')
   getattr(process,"combinatoricRecoTaus"+postfix).builders[0].pfCandSrc = cms.InputTag('pfNoPileUpForBoostedTaus')
   getattr(process,"combinatoricRecoTaus"+postfix).modifiers.remove(getattr(process,"combinatoricRecoTaus"+postfix).modifiers[3])

   getattr(process,"hpsPFTauDiscriminationByLooseMuonRejection3"+postfix).dRmuonMatch = cms.double(0.3)
   getattr(process,"hpsPFTauDiscriminationByLooseMuonRejection3"+postfix).dRmuonMatchLimitedToJetArea = cms.bool(True)
   getattr(process,"hpsPFTauDiscriminationByTightMuonRejection3"+postfix).dRmuonMatch = cms.double(0.3)
   getattr(process,"hpsPFTauDiscriminationByTightMuonRejection3"+postfix).dRmuonMatchLimitedToJetArea = cms.bool(True)

