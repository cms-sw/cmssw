import FWCore.ParameterSet.Config as cms

from PhysicsTools.SelectorUtils.tools.DataFormat import DataFormat

def loadEgmIdSequence(process, dataFormat):
    process.load("RecoEgamma.PhotonIdentification.egmPhotonIDs_cfi")
    from PhysicsTools.SelectorUtils.centralIDRegistry import central_id_registry

    # Load the producer for MVA IDs. Make sure it is also added to the sequence!
    process.load("RecoEgamma.PhotonIdentification.PhotonMVAValueMapProducer_cfi")
    process.egmPhotonIDTask = cms.Task()
    # Add everything else other then isolation
    process.egmPhotonIDTask.add(process.photonMVAValueMapProducer,
                                process.egmPhotonIDs)

    process.egmPhotonIDSequence = cms.Sequence(process.egmPhotonIDTask)
