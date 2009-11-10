import FWCore.ParameterSet.Config as cms

import RecoJets.JetProducers.ak5PFJets_cfi
import RecoJets.JetProducers.ic5PFJets_cfi

def jetAlgo( algo ):
    
    print 'PF2PAT: selecting jet algorithm ', algo
    
    if algo == 'IC5':
#allPfJets = RecoJets.JetProducers.ic5PFJets_cfi.iterativeCone5PFJets.clone()
        jetAlgo = RecoJets.JetProducers.ic5PFJets_cfi.iterativeCone5PFJets.clone()
    elif algo == 'AK5':
        jetAlgo = RecoJets.JetProducers.ak5PFJets_cfi.ak5PFJets.clone()
        
    jetAlgo.src = 'pfNoElectron'
    return jetAlgo    
