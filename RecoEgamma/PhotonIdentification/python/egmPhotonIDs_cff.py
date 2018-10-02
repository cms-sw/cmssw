import FWCore.ParameterSet.Config as cms

from PhysicsTools.SelectorUtils.tools.DataFormat import DataFormat

def loadEgmIdSequence(process, dataFormat):
    process.load("RecoEgamma.PhotonIdentification.egmPhotonIDs_cfi")
    from PhysicsTools.SelectorUtils.centralIDRegistry import central_id_registry

    # Load the producer module to build full 5x5 cluster shapes and whatever 
    # else is needed for IDs
    process.load("RecoEgamma.PhotonIdentification.photonIDValueMapProducer_cff")

    # Load the producer for MVA IDs. Make sure it is also added to the sequence!
    process.load("RecoEgamma.PhotonIdentification.PhotonMVAValueMapProducer_cfi")

    # Load tasks for isolations computed with CITK for both AOD and miniAOD cases
    process.egmPhotonIDTask = cms.Task()
    # The isolation piece is different depending on the input format
    if dataFormat== DataFormat.AOD:
        process.load("RecoEgamma.EgammaIsolationAlgos.egmPhotonIsolationAOD_cff")
        #if particleFlowTmpPtrs was not create we should create it
        if not hasattr(process, "particleFlowTmpPtrs"):
            process.particleFlowTmpPtrs = cms.EDProducer("PFCandidateFwdPtrProducer",
                                                         src = cms.InputTag('particleFlow')
                                                         )
            process.egmPhotonIDTask.add(process.particleFlowTmpPtrs,
                                        process.egmPhotonIsolationAODTask)
        else :
            process.egmPhotonIDTask.add(process.egmPhotonIsolationAODTask)
            
    elif dataFormat== DataFormat.MiniAOD:
        process.load("RecoEgamma.EgammaIsolationAlgos.egmPhotonIsolationMiniAOD_cff")
        process.egmPhotonIDTask.add(process.egmPhotonIsolationMiniAODTask)
    else:
        raise Exception('InvalidVIDDataFormat', 'The requested data format is different from AOD or MiniAOD')
    # Add everything else other then isolation
    process.egmPhotonIDTask.add(process.photonIDValueMapProducer,
                                process.photonMVAValueMapProducer,
                                process.egmPhotonIDs)
    process.egmPhotonIDSequence = cms.Sequence(process.egmPhotonIDTask)
