import FWCore.ParameterSet.Config as cms

EcalRawToRecHitRoI = cms.EDProducer("EcalRawToRecHitRoI",
    JetJobPSet = cms.VPSet(),
    CandJobPSet = cms.VPSet(),
    sourceTag = cms.InputTag("EcalRawToRecHitFacility"),
    EmJobPSet = cms.VPSet(),
    type = cms.string('muon egamma jet candidate all'),
    MuonJobPSet = cms.PSet(

    )
)


