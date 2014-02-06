import FWCore.ParameterSet.Config as cms

import RecoJets.JetProducers.ak4PFJets_cfi
import RecoJets.JetProducers.ic5PFJets_cfi

def jetAlgo( algo ):

    # print 'PF2PAT: selecting jet algorithm ', algo

    if algo == 'IC5':
#allPfJets = RecoJets.JetProducers.ic5PFJets_cfi.iterativeCone5PFJets.clone()
        jetAlgo = RecoJets.JetProducers.ic5PFJets_cfi.iterativeCone5PFJets.clone()
    elif algo == 'AK4':
        jetAlgo = RecoJets.JetProducers.ak4PFJets_cfi.ak4PFJets.clone()
    elif algo == 'AK7':
        jetAlgo = RecoJets.JetProducers.ak4PFJets_cfi.ak4PFJets.clone()
        jetAlgo.rParam = cms.double(0.7)
        jetAlgo.doAreaFastjet = cms.bool(False)

    jetAlgo.src = 'pfNoElectronJME'
    jetAlgo.doPVCorrection = True
    jetAlgo.srcPVs = cms.InputTag("goodOfflinePrimaryVertices")
    return jetAlgo
