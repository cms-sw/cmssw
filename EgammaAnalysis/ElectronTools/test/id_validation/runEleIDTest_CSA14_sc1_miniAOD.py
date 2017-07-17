import FWCore.ParameterSet.Config as cms

process = cms.Process("TestID")

process.load("FWCore.MessageService.MessageLogger_cfi")

# Load stuff needed by lazy tools later for the value map producer
process.load("Configuration.StandardSequences.Services_cff")
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.MagneticField_38T_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")
# NOTE: the pick the right global tag!
#    for CSA14 scenario 1: global tag is 'POSTLS170_V6::All'
#    for CSA14 scenario 2: global tag is 'PLS170_V7AN1::All'
#  as a rule, find the global tag in the DAS under the Configs for given dataset
process.GlobalTag.globaltag = 'PLS170_V7AN1::All'

process.load("EgammaAnalysis/ElectronTools/Validation/DYJetsToLL_Sc1_MINIAOD")

process.wp1 = cms.EDAnalyzer('MiniAODElectronIDValidationAnalyzer',
                             electrons = cms.InputTag("slimmedElectrons"),
                             electronIDs = cms.InputTag("cutBasedElectronID-CSA14-50ns-V1-standalone-veto"),
                             vertices = cms.InputTag("offlineSlimmedPrimaryVertices"),
                             genparticles = cms.InputTag("prunedGenParticles"),
                             convcollection = cms.InputTag("reducedEgamma:reducedConversions"),
                             beamspot = cms.InputTag("offlineBeamSpot"),
                             full5x5SigmaIEtaIEtaMap = cms.InputTag("electronIDValueMapProducer:eleFull5x5SigmaIEtaIEta"),
                             )

process.wp2 = cms.EDAnalyzer('MiniAODElectronIDValidationAnalyzer',
                             electrons = cms.InputTag("slimmedElectrons"),
                             electronIDs = cms.InputTag("cutBasedElectronID-CSA14-50ns-V1-standalone-loose"),
                             vertices = cms.InputTag("offlineSlimmedPrimaryVertices"),
                             genparticles = cms.InputTag("prunedGenParticles"),
                             convcollection = cms.InputTag("reducedEgamma:reducedConversions"),
                             beamspot = cms.InputTag("offlineBeamSpot"),
                             full5x5SigmaIEtaIEtaMap = cms.InputTag("electronIDValueMapProducer:eleFull5x5SigmaIEtaIEta"),
                             )

process.wp3 = cms.EDAnalyzer('MiniAODElectronIDValidationAnalyzer',
                             electrons = cms.InputTag("slimmedElectrons"),
                             electronIDs = cms.InputTag("cutBasedElectronID-CSA14-50ns-V1-standalone-medium"),
                             vertices = cms.InputTag("offlineSlimmedPrimaryVertices"),
                             genparticles = cms.InputTag("prunedGenParticles"),
                             convcollection = cms.InputTag("reducedEgamma:reducedConversions"),
                             beamspot = cms.InputTag("offlineBeamSpot"),
                             full5x5SigmaIEtaIEtaMap = cms.InputTag("electronIDValueMapProducer:eleFull5x5SigmaIEtaIEta"),
                             )

process.wp4 = cms.EDAnalyzer('MiniAODElectronIDValidationAnalyzer',
                             electrons = cms.InputTag("slimmedElectrons"),
                             electronIDs = cms.InputTag("cutBasedElectronID-CSA14-50ns-V1-standalone-tight"),
                             vertices = cms.InputTag("offlineSlimmedPrimaryVertices"),
                             genparticles = cms.InputTag("prunedGenParticles"),
                             convcollection = cms.InputTag("reducedEgamma:reducedConversions"),
                             beamspot = cms.InputTag("offlineBeamSpot"),
                             full5x5SigmaIEtaIEtaMap = cms.InputTag("electronIDValueMapProducer:eleFull5x5SigmaIEtaIEta"),
                             )

process.TFileService = cms.Service("TFileService",
                                   fileName = cms.string('electron_ntuple_sc1_miniAOD.root')
                                   )

process.p = cms.Path(process.wp1 * process.wp2 * process.wp3 * process.wp4)

