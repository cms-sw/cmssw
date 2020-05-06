import FWCore.ParameterSet.Config as cms

# AOD content
RecoHiJetsAOD = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep *_ak4CaloJets_*_*',
        'keep *_akPu4CaloJets_*_*', 
        'keep *_ak4PFJets_*_*',
        'keep *_akPu3PFJets_*_*',
        'keep *_akPu4PFJets_*_*',
        'keep *_akCs3PFJets_*_*',
        'keep *_akCs4PFJets_*_*',
        'keep *_PFTowers_*_*',
        'keep *_hiFJRhoProducer_*_*',
        'keep CaloTowersSorted_towerMaker_*_*',
        'keep recoPFCandidates_particleFlowTmp_*_*')
)

# RECO content
RecoHiJetsRECO = cms.PSet(
    outputCommands = cms.untracked.vstring( 
        'keep *_akPu3CaloJets_*_*', 
        'keep *_akPu5CaloJets_*_*', 
        'keep *_iterativeConePu5CaloJets_*_*', 
        'keep *_akPu5PFJets_*_*',
        'keep recoCandidatesOwned_caloTowers_*_*',
        'keep *_kt4PFJetsForRho_*_*')
)
RecoHiJetsRECO.outputCommands.extend(RecoHiJetsAOD.outputCommands)

#Full Event content 
RecoHiJetsFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring(
	'keep *_*CaloJets_*_*',
        'keep *_*PFJets_*_*')
)
RecoHiJetsFEVT.outputCommands.extend(RecoHiJetsRECO.outputCommands)
