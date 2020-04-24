import FWCore.ParameterSet.Config as cms

process = cms.Process("TestID")

process.load("FWCore.MessageService.MessageLogger_cfi")

# Load stuff needed by lazy tools later for the value map producer
process.load("Configuration.StandardSequences.Services_cff")
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.MagneticField_38T_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
#process.load("Configuration.StandardSequences.Reconstruction_cff")
# NOTE: the pick the right global tag!
#    for CSA14 scenario 1: global tag is 'POSTLS170_V6::All'
#    for CSA14 scenario 2: global tag is 'PLS170_V7AN1::All'
#  as a rule, find the global tag in the DAS under the Configs for given dataset
process.GlobalTag.globaltag = 'POSTLS170_V6::All'

process.load("EgammaAnalysis/ElectronTools/Validation/DYJetsToLL_Sc1_AODSIM")

process.source.fileNames = cms.untracked.vstring(
    '/store/relval/CMSSW_7_2_0_pre4/RelValZEE_13/GEN-SIM-RECO/PU50ns_POSTLS172_V4-v3/00000/207152BC-CF27-E411-BC7C-0025905B8562.root',
    '/store/relval/CMSSW_7_2_0_pre4/RelValZEE_13/GEN-SIM-RECO/PU50ns_POSTLS172_V4-v3/00000/207CFD5F-C127-E411-BEDB-0026189438C0.root',
    '/store/relval/CMSSW_7_2_0_pre4/RelValZEE_13/GEN-SIM-RECO/PU50ns_POSTLS172_V4-v3/00000/6807E684-C227-E411-8832-002354EF3BDC.root',
    '/store/relval/CMSSW_7_2_0_pre4/RelValZEE_13/GEN-SIM-RECO/PU50ns_POSTLS172_V4-v3/00000/92A5C477-C427-E411-874E-0025905A6110.root',
    '/store/relval/CMSSW_7_2_0_pre4/RelValZEE_13/GEN-SIM-RECO/PU50ns_POSTLS172_V4-v3/00000/A81E8828-C627-E411-AD32-0025905A60D0.root'
)

from PhysicsTools.SelectorUtils.tools.vid_id_tools import *
# turn on VID producer
switchOnVIDElectronIdProducer(process)
# define which IDs we want to produce
my_id_modules = ['EgammaAnalysis.ElectronTools.Identification.cutBasedElectronID_CSA14_50ns_V1_cff',
                 'EgammaAnalysis.ElectronTools.Identification.cutBasedElectronID_CSA14_PU20bx25_V0_cff',
                 'EgammaAnalysis.ElectronTools.Identification.heepElectronID_HEEPV50_CSA14_25ns_cff',
                 'EgammaAnalysis.ElectronTools.Identification.heepElectronID_HEEPV50_CSA14_startup_cff']
#add them to the VID producer
for idmod in my_id_modules:
    setupAllVIDIdsInModule(process,idmod,setupVIDElectronSelection)

process.electronIDValueMapProducer.ebReducedRecHitCollection = cms.InputTag('reducedEcalRecHitsEB')
process.electronIDValueMapProducer.eeReducedRecHitCollection = cms.InputTag('reducedEcalRecHitsEE')
process.electronIDValueMapProducer.esReducedRecHitCollection = cms.InputTag('reducedEcalRecHitsES')

process.wp1 = cms.EDAnalyzer('ElectronIDValidationAnalyzer',
                             electrons = cms.InputTag("gedGsfElectrons"),
                             electronIDs = cms.InputTag("egmGsfElectronIDs:cutBasedElectronID-CSA14-50ns-V1-standalone-veto"),
                             vertices = cms.InputTag("offlinePrimaryVertices"),
                             genparticles = cms.InputTag("genParticles"),
                             convcollection = cms.InputTag("conversions"),
                             beamspot = cms.InputTag("offlineBeamSpot"),
                             full5x5SigmaIEtaIEtaMap = cms.InputTag("electronIDValueMapProducer:eleFull5x5SigmaIEtaIEta"),
                             )

process.wp2 = cms.EDAnalyzer('ElectronIDValidationAnalyzer',
                             electrons = cms.InputTag("gedGsfElectrons"),
                             electronIDs = cms.InputTag("egmGsfElectronIDs:cutBasedElectronID-CSA14-50ns-V1-standalone-loose"),
                             vertices = cms.InputTag("offlinePrimaryVertices"),
                             genparticles = cms.InputTag("genParticles"),
                             convcollection = cms.InputTag("conversions"),
                             beamspot = cms.InputTag("offlineBeamSpot"),
                             full5x5SigmaIEtaIEtaMap = cms.InputTag("electronIDValueMapProducer:eleFull5x5SigmaIEtaIEta"),
                             )

process.wp3 = cms.EDAnalyzer('ElectronIDValidationAnalyzer',
                             electrons = cms.InputTag("gedGsfElectrons"),
                             electronIDs = cms.InputTag("egmGsfElectronIDs:cutBasedElectronID-CSA14-50ns-V1-standalone-medium"),
                             vertices = cms.InputTag("offlinePrimaryVertices"),
                             genparticles = cms.InputTag("genParticles"),
                             convcollection = cms.InputTag("conversions"),
                             beamspot = cms.InputTag("offlineBeamSpot"),
                             full5x5SigmaIEtaIEtaMap = cms.InputTag("electronIDValueMapProducer:eleFull5x5SigmaIEtaIEta"),
                             )

process.wp4 = cms.EDAnalyzer('ElectronIDValidationAnalyzer',
                             electrons = cms.InputTag("gedGsfElectrons"),
                             electronIDs = cms.InputTag("egmGsfElectronIDs:cutBasedElectronID-CSA14-50ns-V1-standalone-tight"),
                             vertices = cms.InputTag("offlinePrimaryVertices"),
                             genparticles = cms.InputTag("genParticles"),
                             convcollection = cms.InputTag("conversions"),
                             beamspot = cms.InputTag("offlineBeamSpot"),
                             full5x5SigmaIEtaIEtaMap = cms.InputTag("electronIDValueMapProducer:eleFull5x5SigmaIEtaIEta"),
                             )

process.TFileService = cms.Service("TFileService",
                                   fileName = cms.string('electron_ntuple_sc1.root')
                                   )

process.p = cms.Path(process.egmGsfElectronIDSequence * process.wp1 * process.wp2 * process.wp3 * process.wp4)

