import FWCore.ParameterSet.Config as cms

HLT1Photon = cms.EDProducer("PATHLTProducer",
    triggerResults = cms.InputTag("TriggerResults","","HLT"),
    filterName = cms.InputTag("hltL1IsoSinglePhotonTrackIsolFilter","","HLT"),
    triggerName = cms.string('HLT1Photon')
)

HLT1PhotonRelaxed = cms.EDProducer("PATHLTProducer",
    triggerResults = cms.InputTag("TriggerResults","","HLT"),
    filterName = cms.InputTag("hltL1NonIsoSinglePhotonTrackIsolFilter","","HLT"),
    triggerName = cms.string('HLT1PhotonRelaxed')
)

HLT2Photon = cms.EDProducer("PATHLTProducer",
    triggerResults = cms.InputTag("TriggerResults","","HLT"),
    filterName = cms.InputTag("hltL1IsoDoublePhotonDoubleEtFilter","","HLT"),
    triggerName = cms.string('HLT2Photon')
)

HLT2PhotonRelaxed = cms.EDProducer("PATHLTProducer",
    triggerResults = cms.InputTag("TriggerResults","","HLT"),
    filterName = cms.InputTag("hltL1NonIsoDoublePhotonDoubleEtFilter","","HLT"),
    triggerName = cms.string('HLT2PhotonRelaxed')
)

photonHLTProducer = cms.Sequence(HLT1Photon*HLT1PhotonRelaxed*HLT2Photon*HLT2PhotonRelaxed)

