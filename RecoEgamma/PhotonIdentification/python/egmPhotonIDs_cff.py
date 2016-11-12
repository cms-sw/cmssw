import FWCore.ParameterSet.Config as cms

from PhysicsTools.SelectorUtils.tools.auxiliary_cff import DataFormat

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
    if dataFormat== DataFormat.AOD:
        process.load("RecoEgamma.EgammaIsolationAlgos.egmPhotonIsolationAOD_cff")
        process.egmPhotonIDSequence = cms.Sequence(process.egmPhotonIsolationAODSequence 
                                                   * process.photonIDValueMapProducer 
                                                   * process.photonMVAValueMapProducer 
                                                   * process.egmPhotonIDs 
                                                   * process.photonRegressionValueMapProducer )
    elif dataFormat== DataFormat.MiniAOD:
        process.load("RecoEgamma.EgammaIsolationAlgos.egmPhotonIsolationMiniAOD_cff")
        process.egmPhotonIDSequence = cms.Sequence(process.egmPhotonIsolationMiniAODSequence 
                                                   * process.photonIDValueMapProducer 
                                                   * process.photonMVAValueMapProducer 
                                                   * process.egmPhotonIDs 
                                                   * process.photonRegressionValueMapProducer )
    else:
        raise Exception('InvalidVIDDataFormat', 'The requested data format is different from AOD or MiniAOD')
