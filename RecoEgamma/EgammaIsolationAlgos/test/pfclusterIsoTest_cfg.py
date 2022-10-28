import FWCore.ParameterSet.Config as cms

process = cms.Process("eleIso")
process.load("Configuration.StandardSequences.GeometryDB_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'MCRUN2_74_V7'

process.load("Configuration.EventContent.EventContent_cff")
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        '/store/relval/CMSSW_7_4_0_pre8/RelValZEE_13/GEN-SIM-RECO/PU25ns_MCRUN2_74_V7-v1/00000/A4DAB081-55BD-E411-9490-0025905A608E.root',

    )
)
process.out = cms.OutputModule("PoolOutputModule",
                               process.FEVTSIMEventContent,
                               fileName = cms.untracked.string('file:eleIso.root')
)

process.out.outputCommands.append('keep *')
#process.out.outputCommands.append('keep *_gsfElectrons_*_*')
#process.out.outputCommands.append('keep *_photons_*_*')
#process.out.outputCommands.append('keep *_*_*_eleIso')
#process.out.outputCommands.append('keep EcalRecHitsSorted_*_*_*')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(200)
)

process.load("RecoParticleFlow.PFClusterProducer.particleFlowCluster_cff")
process.load("RecoLocalCalo.HcalRecAlgos.hcalRecAlgoESProd_cfi")
process.particleFlowRecHitECAL.producers[0].src=cms.InputTag("reducedEcalRecHitsEB")
process.particleFlowRecHitECAL.producers[1].src=cms.InputTag("reducedEcalRecHitsEE")
process.particleFlowRecHitPS.producers[0].src=cms.InputTag("reducedEcalRecHitsES")
process.particleFlowRecHitHBHE.producers[0].src=cms.InputTag("reducedHcalRecHits","hbhereco") 

process.load("RecoEgamma.EgammaIsolationAlgos.pfClusterIsolation_cfi")

process.p1 = cms.Path(
#    process.particleFlowClusterWithoutHO +
    process.pfClusterIsolationSequence
)

process.outpath = cms.EndPath(process.out)
