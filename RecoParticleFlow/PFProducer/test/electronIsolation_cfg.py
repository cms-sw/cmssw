import FWCore.ParameterSet.Config as cms

process = cms.Process("ELECISO")


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(2000)
)

process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(
    '/store/relval/CMSSW_4_4_0_pre3/RelValQCD_FlatPt_15_3000/GEN-SIM-RECO/START43_V4-v1/0000/0EE664B2-FFA3-E011-918F-002618943882.root',
    '/store/relval/CMSSW_4_4_0_pre3/RelValQCD_FlatPt_15_3000/GEN-SIM-RECO/START43_V4-v1/0000/644190E4-F0A3-E011-BED0-00304867C1BA.root',
    '/store/relval/CMSSW_4_4_0_pre3/RelValQCD_FlatPt_15_3000/GEN-SIM-RECO/START43_V4-v1/0000/9A80E178-18A4-E011-B1DD-002618FDA287.root',
    '/store/relval/CMSSW_4_4_0_pre3/RelValQCD_FlatPt_15_3000/GEN-SIM-RECO/START43_V4-v1/0005/F29B164E-43A6-E011-B2B1-00248C55CC7F.root'    )
)


# path ---------------------------------------------------------------

process.dump = cms.EDAnalyzer("EventContentAnalyzer")

process.load("RecoParticleFlow.PFProducer.pfBasedElectronIso_cff")
process.load("RecoParticleFlow.PFProducer.pfBasedPhotonIso_cff")
process.load("RecoParticleFlow.PFProducer.pfLinker_cfi")
process.pfLinker.ProducePFCandidates = cms.bool(False)
process.pfLinker.PFCandidate = [cms.InputTag('pfSelectedPhotons'), cms.InputTag('pfSelectedElectrons')]
process.pfLinker.FillMuonRefs = cms.bool(False)
#process.pfPileUpCandidates.bottomCollection = cms.InputTag('particleFlow')

process.isoReader = cms.EDAnalyzer("PFIsoReader",
                                   PFCandidates = cms.InputTag('pfSelectedPhotons'),
                                   Electrons=cms.InputTag('gsfElectrons'),
                                   Photons=cms.InputTag('pfPhotonTranslator:pfphot'),
                                   ElectronValueMap=cms.InputTag('pfLinker:electrons'),
                                   PhotonValueMap=cms.InputTag('pfLinker:photons'),
                                   MergedValueMap=cms.InputTag('pfLinker:all'),
                                   ElectronIsoDeposits = cms.VInputTag(cms.InputTag('elPFIsoDepositCharged'),
                                                                       cms.InputTag('elPFIsoDepositNeutral'),
                                                                       cms.InputTag('elPFIsoDepositGamma')),
                                   PhotonIsoDeposits = cms.VInputTag(cms.InputTag('phPFIsoDepositCharged'),
                                                                     cms.InputTag('phPFIsoDepositNeutral'),
                                                                     cms.InputTag('phPFIsoDepositGamma')),
                                   useEGPFValueMaps=cms.bool(True))
                                                          

process.load("FWCore.Modules.printContent_cfi")
process.p = cms.Path(
    process.pfBasedElectronIsoSequence+
    process.pfBasedPhotonIsoSequence+
    process.pfLinker+
    process.isoReader
    )

# output ------------------------------------------------------------

process.load("Configuration.EventContent.EventContent_cff")

process.out = cms.OutputModule("PoolOutputModule",
                               outputCommands = cms.untracked.vstring('drop *',
                                                                      'keep *_*_*_ELECISO'),
                               fileName = cms.untracked.string('electronIsolation.root')
)



process.outpath = cms.EndPath(
    process.out 
#    process.aod
    )


# other stuff

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 10

