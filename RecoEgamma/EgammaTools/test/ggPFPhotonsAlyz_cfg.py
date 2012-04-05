import FWCore.ParameterSet.Config as cms

process = cms.Process("Demo")

process.load('Configuration/StandardSequences/Services_cff')
process.load('FWCore/MessageService/MessageLogger_cfi')
process.load('Configuration.StandardSequences.GeometryDB_cff')
process.load('Configuration/StandardSequences/MagneticField_38T_cff')
process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
process.load('Configuration/EventContent/EventContent_cff')

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.source = cms.Source("PoolSource",
   fileNames = cms.untracked.vstring('/store/relval/CMSSW_6_0_0_pre1/RelValH130GGgluonfusion/GEN-SIM-RECO/START52_V4-v1/0117/40ACF8F2-AE77-E111-8ECA-002618943962.root'),
# fileNames = cms.untracked.vstring('/store/relval/CMSSW_6_0_0_pre1/RelValSingleElectronPt35/GEN-SIM-RECO/START52_V4-v1/0118/60EBD233-0C78-E111-A649-003048FFCBA4.root'),                           
)

process.ggPFPhotonAnalyzer = cms.EDAnalyzer('ggPFPhotonAnalyzer',
                                            PFParticles =cms.InputTag("particleFlow"),
                                            PFPhotons = cms.InputTag("pfPhotonTranslator:pfphot"),
                                            PFElectrons = cms.InputTag("gsfElectrons"),
                                            Photons=cms.InputTag("photons"),
                                            ebReducedRecHitCollection = cms.InputTag("reducedEcalRecHitsEB"),
                                            eeReducedRecHitCollection = cms.InputTag("reducedEcalRecHitsEE"),
                                            esRecHitCollection = cms.InputTag("reducedEcalRecHitsES"),
                                            BeamSpotCollection = cms.InputTag("offlineBeamSpot")
                                            )

process.GlobalTag.globaltag = 'START52_V4::All'

process.p = cms.Path(process.ggPFPhotonAnalyzer)
