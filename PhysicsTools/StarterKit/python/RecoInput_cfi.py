import FWCore.ParameterSet.Config as cms

# from 

def RecoInput() : 
 return cms.Source("PoolSource",
                   debugVerbosity = cms.untracked.uint32(200),
                   debugFlag = cms.untracked.bool(True),
                   
                   fileNames = cms.untracked.vstring(
	'/store/relval/CMSSW_2_2_8/RelValZMM/GEN-SIM-RECO/STARTUP_V9_v1/0000/6820DE4B-BE2C-DE11-975D-000423D99658.root',
        '/store/relval/CMSSW_2_2_8/RelValZMM/GEN-SIM-RECO/STARTUP_V9_v1/0000/7CB3E361-BA2C-DE11-8D51-001617C3B710.root',
        '/store/relval/CMSSW_2_2_8/RelValZMM/GEN-SIM-RECO/STARTUP_V9_v1/0000/80069F07-B72C-DE11-BECB-000423D98E54.root',
	'/store/relval/CMSSW_2_2_8/RelValZMM/GEN-SIM-RECO/STARTUP_V9_v1/0000/E81D418B-E42C-DE11-8125-001D09F24259.root'

     )
                   )



