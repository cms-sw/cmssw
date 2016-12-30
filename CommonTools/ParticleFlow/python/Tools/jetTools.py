import FWCore.ParameterSet.Config as cms

def jetAlgo( algo ):

    # print 'PF2PAT: selecting jet algorithm ', algo

    if algo == 'IC5':
        import RecoJets.JetProducers.ic5PFJets_cfi as jetConfig
        jetAlgo = jetConfig.iterativeCone5PFJets.clone()
    elif algo == 'AK4':
        import RecoJets.JetProducers.ak4PFJets_cfi as jetConfig
        jetAlgo = jetConfig.ak4PFJets.clone()
    elif algo == 'AK7':
        import RecoJets.JetProducers.ak4PFJets_cfi as jetConfig
        jetAlgo = jetConfig.ak4PFJets.clone()
        jetAlgo.rParam = cms.double(0.7)
        jetAlgo.doAreaFastjet = cms.bool(False)

    jetAlgo.src = 'pfNoElectronJME'
    jetAlgo.doPVCorrection = True
    jetAlgo.srcPVs = cms.InputTag("goodOfflinePrimaryVertices")
    return jetAlgo
