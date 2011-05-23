# official example for PF2PAT

import FWCore.ParameterSet.Config as cms

process = cms.Process("ELECISO")


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(2000)
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
       '/store/relval/CMSSW_4_2_3/RelValH130GGgluonfusion/GEN-SIM-RECO/START42_V12-v2/0067/0A2A882D-087C-E011-B6E4-00248C0BE016.root',
       '/store/relval/CMSSW_4_2_3/RelValH130GGgluonfusion/GEN-SIM-RECO/START42_V12-v2/0063/7E911D11-B47B-E011-A719-001A92971AD8.root',
       '/store/relval/CMSSW_4_2_3/RelValH130GGgluonfusion/GEN-SIM-RECO/START42_V12-v2/0063/12BDD2DE-857B-E011-8019-002618943906.root',
       '/store/relval/CMSSW_4_2_3/RelValH130GGgluonfusion/GEN-SIM-RECO/START42_V12-v2/0062/62E92439-357B-E011-A3FF-001A928116D0.root',
       '/store/relval/CMSSW_4_2_3/RelValH130GGgluonfusion/GEN-SIM-RECO/START42_V12-v2/0062/1E08A9B0-2C7B-E011-905B-00261894395F.root'
       )
)


# path ---------------------------------------------------------------

process.dump = cms.EDAnalyzer("EventContentAnalyzer")

process.load("RecoParticleFlow.PFProducer.pfBasedElectronIso_cff")
process.load("RecoParticleFlow.PFProducer.pfBasedPhotonIso_cff")
process.load("RecoParticleFlow.PFProducer.pfGsfElectronLinker_cfi")
process.egammaLinker.ProducePFCandidates = False

process.isoReader = cms.EDAnalyzer("PFIsoReader",
                                   PFCandidates = cms.InputTag('pfSelectedPhotons'),
                                   Electrons=cms.InputTag('gsfElectrons'),
                                   Photons=cms.InputTag('pfPhotonTranslator:pfphot'),
                                   ElectronValueMap=cms.InputTag('egammaLinker:electrons'),
                                   PhotonValueMap=cms.InputTag('egammaLinker:photons'),
                                   ElectronIsoDeposits = cms.VInputTag(cms.InputTag('isoDepElectronWithCharged'),
                                                                       cms.InputTag('isoDepElectronWithPhotons'),
                                                                       cms.InputTag('isoDepElectronWithNeutral')),
                                   PhotonIsoDeposits = cms.VInputTag(cms.InputTag('isoDepPhotonWithCharged'),
                                                                     cms.InputTag('isoDepPhotonWithPhotons'),
                                                                     cms.InputTag('isoDepPhotonWithNeutral')),
                                   useEGPFValueMaps=cms.bool(True))
                                                          

process.p = cms.Path(
    process.pfBasedElectronIsoSequence+
    process.pfBasedPhotonIsoSequence+
    process.egammaLinker+
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
