# official example for PF2PAT

import FWCore.ParameterSet.Config as cms

process = cms.Process("ELECISO")


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(2000)
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
    '/store/relval/CMSSW_4_3_0_pre6/RelValH130GGgluonfusion/GEN-SIM-RECO/START43_V3-v1/0086/80467D34-378C-E011-A37C-0018F3D09658.root',
    '/store/relval/CMSSW_4_3_0_pre6/RelValH130GGgluonfusion/GEN-SIM-RECO/START43_V3-v1/0086/50787FD7-AE8B-E011-AC2B-002618943880.root',
    '/store/relval/CMSSW_4_3_0_pre6/RelValH130GGgluonfusion/GEN-SIM-RECO/START43_V3-v1/0082/362C873C-808B-E011-A576-0018F3D095EC.root',
    '/store/relval/CMSSW_4_3_0_pre6/RelValH130GGgluonfusion/GEN-SIM-RECO/START43_V3-v1/0080/92A7BBA9-B78A-E011-B40A-002618FDA248.root'
    )
)


# path ---------------------------------------------------------------

process.dump = cms.EDAnalyzer("EventContentAnalyzer")

process.load("RecoParticleFlow.PFProducer.pfBasedElectronIso_cff")
process.load("RecoParticleFlow.PFProducer.pfBasedPhotonIso_cff")
process.load("RecoParticleFlow.PFProducer.pfLinker_cfi")
process.pfLinker.ProducePFCandidates = cms.bool(False)
process.pfLinker.PFCandidates = [cms.InputTag('pfSelectedPhotons'), cms.InputTag('pfSelectedElectrons')]
process.pfLinker.FillMuonRefs = cms.bool(False)

process.isoReader = cms.EDAnalyzer("PFIsoReader",
                                   PFCandidates = cms.InputTag('pfSelectedPhotons'),
                                   Electrons=cms.InputTag('gsfElectrons'),
                                   Photons=cms.InputTag('pfPhotonTranslator:pfphot'),
                                   ElectronValueMap=cms.InputTag('pfLinker:electrons'),
                                   PhotonValueMap=cms.InputTag('pfLinker:photons'),
                                   MergedValueMap=cms.InputTag('pfLinker:all'),
                                   ElectronIsoDeposits = cms.VInputTag(cms.InputTag('isoDepElectronWithCharged'),
                                                                       cms.InputTag('isoDepElectronWithPhotons'),
                                                                       cms.InputTag('isoDepElectronWithNeutral')),
                                   PhotonIsoDeposits = cms.VInputTag(cms.InputTag('isoDepPhotonWithCharged'),
                                                                     cms.InputTag('isoDepPhotonWithPhotons'),
                                                                     cms.InputTag('isoDepPhotonWithNeutral')),
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

# the following are necessary for taus:

#process.load("Configuration.StandardSequences.GeometryPilot2_cff")
#process.load("Configuration.StandardSequences.MagneticField_cff")
#process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")

# do not forget to set the global tag according to the
# release you are using and the sample you are reading (data or MC)
# global tags can be found here:
# https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideFrontierConditions#Valid_Global_Tags_by_Release
#process.GlobalTag.globaltag = cms.string('GR09_R_34X_V2::All')
