import FWCore.ParameterSet.Config as cms

EcalRawToRecHitRoI = cms.EDProducer("EcalRawToRecHitRoI",
    JetJobPSet = cms.VPSet(),
    CandJobPSet = cms.VPSet(),
    sourceTag = cms.InputTag("EcalRawToRecHitFacility"),
    EmJobPSet = cms.VPSet(),
    type = cms.string('muon egamma jet candidate all'),
    MuonJobPSet = cms.PSet(

    ),
    doES = cms.bool(False),
    sourceTag_es = cms.InputTag(''),
    esInstance = cms.untracked.string('es')
)


