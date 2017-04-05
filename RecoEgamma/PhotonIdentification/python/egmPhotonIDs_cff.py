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

    # Load tasks for isolations computed with CITK for both AOD and miniAOD cases
    if dataFormat== DataFormat.AOD:
        process.load("RecoEgamma.EgammaIsolationAlgos.egmPhotonIsolationAOD_cff")
        #if particleFlowTmpPtrs was not create we should create it
        if not hasattr(process, "particleFlowTmpPtrs"):
            process.particleFlowTmpPtrs = cms.EDProducer("PFCandidateFwdPtrProducer",
                                                         src = cms.InputTag('particleFlow')
                                                         )
            process.egmPhotonIDTask = cms.Task()
            process.egmPhotonIDTask.add(process.particleFlowTmpPtrs)
            process.egmPhotonIDTask.add(process.egmPhotonIsolationAODTask)
            process.egmPhotonIDTask.add(process.photonIDValueMapProducer )
            process.egmPhotonIDTask.add(process.photonMVAValueMapProducer )
            process.egmPhotonIDTask.add(process.egmPhotonIDs )
            process.egmPhotonIDTask.add(process.photonRegressionValueMapProducer )
        else :
            process.egmPhotonIDTask = cms.Task()
            process.egmPhotonIDTask.add(process.egmPhotonIsolationAODTask)
            process.egmPhotonIDTask.add(process.photonIDValueMapProducer)
            process.egmPhotonIDTask.add(process.photonMVAValueMapProducer)
            process.egmPhotonIDTask.add(process.egmPhotonIDs)
            process.egmPhotonIDTask.add(process.photonRegressionValueMapProducer)
            
    elif dataFormat== DataFormat.MiniAOD:
        process.load("RecoEgamma.EgammaIsolationAlgos.egmPhotonIsolationMiniAOD_cff")
        process.egmPhotonIDTask = cms.Task()
        process.egmPhotonIDTask.add(process.egmPhotonIsolationMiniAODTask)
        process.egmPhotonIDTask.add(process.photonIDValueMapProducer)
        process.egmPhotonIDTask.add(process.photonMVAValueMapProducer)
        process.egmPhotonIDTask.add(process.egmPhotonIDs)
        process.egmPhotonIDTask.add(process.photonRegressionValueMapProducer)
    else:
        raise Exception('InvalidVIDDataFormat', 'The requested data format is different from AOD or MiniAOD')
    # For all cases above, package the task into a sequence
    process.egmPhotonIDSequence = cms.Sequence(process.egmPhotonIDTask)
