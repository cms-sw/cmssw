import FWCore.ParameterSet.Config as cms

from DQMOffline.EGamma.photonAnalyzer_cfi import photonAnalysis as cosmicPhotonAnalysis 

cosmicPhotonAnalysis.minPhoEtCut = cms.double(0.0)
cosmicPhotonAnalysis.eMax = cms.double(3.0)
cosmicPhotonAnalysis.etMax = cms.double(3.0)
cosmicPhotonAnalysis.r9Max = cms.double(1.5)    
