import FWCore.ParameterSet.Config as cms

from PhysicsTools.SelectorUtils.tools.DataFormat import DataFormat

def LoadEgmIdSequence(process, dataFormat):
    process.load("RecoEgamma.PhotonIdentification.egmPhotonIDs_cfi")
    from PhysicsTools.SelectorUtils.centralIDRegistry import central_id_registry

    # Load the producer module to build full 5x5 cluster shapes and whatever 
    # else is needed for IDs
    process.load("RecoEgamma.PhotonIdentification.PhotonIDValueMapProducer_cfi")

    # Load the producer for MVA IDs. Make sure it is also added to the sequence!
    process.load("RecoEgamma.PhotonIdentification.PhotonMVAValueMapProducer_cfi")
    process.load("RecoEgamma.PhotonIdentification.PhotonRegressionValueMapProducer_cfi")
    
    # Load sequences for isolations computed with CITK for both AOD and miniAOD cases
    process.egmPhotonIDSequence = cms.Sequence()
    # The isolation piece is different depending on the input format
    if dataFormat== DataFormat.AOD:
        process.load("RecoEgamma.EgammaIsolationAlgos.egmPhotonIsolationAOD_cff")
        #if particleFlowTmpPtrs was not create we should create it
        if not hasattr(process, "particleFlowTmpPtrs"):
            process.particleFlowTmpPtrs = cms.EDProducer("PFCandidateFwdPtrProducer",
                                                         src = cms.InputTag('particleFlow')
                                                         )
            isoSequence = cms.Sequence(process.particleFlowTmpPtrs +
                                       process.egmPhotonIsolationAODSequence)
        else :
            isoSequence = cms.Sequence(process.egmPhotonIsolationAODSequence)

    elif dataFormat== DataFormat.MiniAOD:
        process.load("RecoEgamma.EgammaIsolationAlgos.egmPhotonIsolationMiniAOD_cff")
        isoSequence = cms.Sequence(process.egmPhotonIsolationMiniAODSequence)
    else:
        raise Exception('InvalidVIDDataFormat', 'The requested data format is different from AOD or MiniAOD')
    # Add everything together
    process.egmPhotonIDSequence = cms.Sequence( isoSequence + 
                                                process.photonIDValueMapProducer +
                                                process.photonMVAValueMapProducer +
                                                process.egmPhotonIDs +
                                                process.photonRegressionValueMapProducer)
