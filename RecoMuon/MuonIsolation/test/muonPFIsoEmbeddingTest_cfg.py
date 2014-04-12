import FWCore.ParameterSet.Config as cms

process = cms.Process("IsoMuon")
# Messages
#process.load("RecoMuon.Configuration.MessageLogger_cfi")
process.load("FWCore.MessageService.MessageLogger_cfi")

# Muon Isolation

process.load("RecoMuon.MuonIsolation.muonPFIsolation_cff")



process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(
    'file:/tmp/bachtis/test.root'
  
                                )
                            )
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)


process.pfEmbeddedMuons = cms.EDProducer("MuPFIsoEmbedder",
                                         src = cms.InputTag("muons"),
                                         isolationR03 = cms.PSet(
                                             chargedParticle = cms.InputTag("muPFIsoValueChargedAll03"),
                                             chargedHadron = cms.InputTag("muPFIsoValueCharged03"),
                                             neutralHadron = cms.InputTag("muPFIsoValueNeutral03"),
                                             photon = cms.InputTag("muPFIsoValueGamma03"),
                                             neutralHadronHighThreshold = cms.InputTag("muPFIsoValueNeutralHighThreshold03"),
                                             photonHighThreshold = cms.InputTag("muPFIsoValueGammaHighThreshold03"),
                                             pu = cms.InputTag("muPFIsoValuePU03")
                                         ),
                                         isolationR04 = cms.PSet(
                                             chargedParticle = cms.InputTag("muPFIsoValueChargedAll04"),
                                             chargedHadron = cms.InputTag("muPFIsoValueCharged04"),
                                             neutralHadron = cms.InputTag("muPFIsoValueNeutral04"),
                                             photon = cms.InputTag("muPFIsoValueGamma04"),
                                             neutralHadronHighThreshold = cms.InputTag("muPFIsoValueNeutralHighThreshold04"),
                                             photonHighThreshold = cms.InputTag("muPFIsoValueGammaHighThreshold04"),
                                             pu = cms.InputTag("muPFIsoValuePU04")
                                         )
)                                         
                                         



process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('IsoMuons.root')
)

process.p = cms.Path(
    process.muonPrePFIsolationSequence+
    process.muonPFIsolationSequence+
    process.pfEmbeddedMuons    
    )

process.this_is_the_end = cms.EndPath(process.out)




