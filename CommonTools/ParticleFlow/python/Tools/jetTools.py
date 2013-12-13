import FWCore.ParameterSet.Config as cms

import RecoJets.JetProducers.ak4PFJets_cfi

def jetAlgo( algo ):
    
    # print 'PF2PAT: selecting jet algorithm ', algo
    
    if algo == 'AK4':
        jetAlgo = RecoJets.JetProducers.ak4PFJets_cfi.ak4PFJets.clone()
    elif algo == 'AK8':
        jetAlgo = RecoJets.JetProducers.ak4PFJets_cfi.ak4PFJets.clone()    
        jetAlgo.rParam = cms.double(0.8)
        jetAlgo.doAreaFastjet = cms.bool(False)
        
    jetAlgo.src = 'pfNoElectronJME'
    jetAlgo.doPVCorrection = True
    jetAlgo.srcPVs = cms.InputTag("goodOfflinePrimaryVertices")
    return jetAlgo    
