import FWCore.ParameterSet.Config as cms
# AOD content
RecoHiCentralityAOD = cms.PSet(
    outputCommands = cms.untracked.vstring(
	'keep recoCentrality*_hiCentrality_*_*',
        'keep *_centralityBin_*_*',
        'keep recoClusterCompatibility*_hiClusterCompatibility_*_*')
)

# RECO content
RecoHiCentralityRECO = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
RecoHiCentralityRECO.outputCommands.extend(RecoHiCentralityAOD.outputCommands)

# FEVT content
RecoHiCentralityFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
RecoHiCentralityFEVT.outputCommands.extend(RecoHiCentralityRECO.outputCommands)
