import FWCore.ParameterSet.Config as cms
process = cms.Process("RECO2")

process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.Geometry.GeometrySimDB_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_PostLS1_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load('Configuration.StandardSequences.RawToDigi_cff')
process.load('Configuration.StandardSequences.Reconstruction_cff')

process.GlobalTag.globaltag = 'START72_V1::All'
process.GlobalTag.toGet = cms.VPSet(
    cms.PSet(record = cms.string("GeometryFileRcd"),
             tag = cms.string("XMLFILE_Geometry_2015_72YV2_Extended2015ZeroMaterial_mc"),
             connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_GEOMETRY_000"),
             label = cms.untracked.string("Extended2015ZeroMaterial")
             )
    )

#### CONFIGURE IT HERE
isMC = True
runOnRAW = False
useTrivial = True
#####################
process.MessageLogger.cerr.FwkReport.reportEvery = 1

# get timing service up for profiling
# process.TimerService = cms.Service("TimerService")
# process.options = cms.untracked.PSet(
#     wantSummary = cms.untracked.bool(True)
# )

process.raw2digi_step = cms.Sequence(process.RawToDigi)

# get uncalibrechits with weights method
import RecoLocalCalo.EcalRecProducers.ecalWeightUncalibRecHit_cfi
process.ecalWeightsUncalibRecHit = RecoLocalCalo.EcalRecProducers.ecalWeightUncalibRecHit_cfi.ecalWeightUncalibRecHit.clone()

# get uncalib rechits from multifit method
import RecoLocalCalo.EcalRecProducers.ecalMultiFitUncalibRecHit_cfi
process.ecalMultiFitUncalibRecHit =  RecoLocalCalo.EcalRecProducers.ecalMultiFitUncalibRecHit_cfi.ecalMultiFitUncalibRecHit.clone()

# get rechits e.g. from the weights
process.load("CalibCalorimetry.EcalLaserCorrection.ecalLaserCorrectionService_cfi")
process.load("RecoLocalCalo.EcalRecProducers.ecalRecHit_cfi")

process.ecalRecHit.triggerPrimitiveDigiCollection = 'ecalEBunpacker:EcalTriggerPrimitives'

# get the recovered digis
if isMC:
    process.ecalDetIdToBeRecovered.ebSrFlagCollection = 'simEcalDigis:ebSrFlags'
    process.ecalDetIdToBeRecovered.eeSrFlagCollection = 'simEcalDigis:eeSrFlags'
    process.ecalRecHit.recoverEBFE = False
    process.ecalRecHit.recoverEEFE = False
    process.ecalRecHit.killDeadChannels = False
if runOnRAW:
    process.ecalDetIdToBeRecovered.ebIntegrityGainErrors = 'ecalEBunpacker:EcalIntegrityGainErrors'
    process.ecalDetIdToBeRecovered.ebIntegrityGainSwitchErrors = 'ecalEBunpacker:EcalIntegrityGainSwitchErrors'
    process.ecalDetIdToBeRecovered.ebIntegrityChIdErrors = 'ecalEBunpacker:EcalIntegrityChIdErrors'
    process.ecalDetIdToBeRecovered.eeIntegrityGainErrors = 'ecalEBunpacker:EcalIntegrityGainErrors'
    process.ecalDetIdToBeRecovered.eeIntegrityGainSwitchErrors = 'ecalEBunpacker:EcalIntegrityGainSwitchErrors'
    process.ecalDetIdToBeRecovered.eeIntegrityChIdErrors = 'ecalEBunpacker:EcalIntegrityChIdErrors'
    process.ecalDetIdToBeRecovered.integrityTTIdErrors = 'ecalEBunpacker:EcalIntegrityTTIdErrors'
    process.ecalDetIdToBeRecovered.integrityBlockSizeErrors = 'ecalEBunpacker:EcalIntegrityBlockSizeErrors'
else:
    process.ecalRecHit.ebDetIdToBeRecovered = ''
    process.ecalRecHit.eeDetIdToBeRecovered = ''
    process.ecalRecHit.ebFEToBeRecovered = ''
    process.ecalRecHit.eeFEToBeRecovered = ''


process.ecalRecHitWeights = process.ecalRecHit.clone()
process.ecalRecHitWeights.EBuncalibRecHitCollection = 'ecalWeightsUncalibRecHit:EcalUncalibRecHitsEB'
process.ecalRecHitWeights.EEuncalibRecHitCollection = 'ecalWeightsUncalibRecHit:EcalUncalibRecHitsEE'
process.ecalRecHitWeights.EBrechitCollection = 'EcalRecHitsWeightsEB'
process.ecalRecHitWeights.EErechitCollection = 'EcalRecHitsWeightsEE'

process.ecalRecHitMultiFit = process.ecalRecHit.clone()
process.ecalRecHitMultiFit.EBuncalibRecHitCollection = 'ecalMultiFitUncalibRecHit:EcalUncalibRecHitsEB'
process.ecalRecHitMultiFit.EEuncalibRecHitCollection = 'ecalMultiFitUncalibRecHit:EcalUncalibRecHitsEE'
process.ecalRecHitMultiFit.EBrechitCollection = 'EcalRecHitsMultiFitEB'
process.ecalRecHitMultiFit.EErechitCollection = 'EcalRecHitsMultiFitEE'

process.maxEvents = cms.untracked.PSet(  input = cms.untracked.int32(100) )
process.source = cms.Source("PoolSource",
              fileNames = cms.untracked.vstring('file:/afs/cern.ch/work/e/emanuele/ecalreco/generate-720pre4-slc6/CMSSW_7_2_0_pre4/src/crab/pu40_25ns/photongun_pu25.root')
                ) 


process.out = cms.OutputModule("PoolOutputModule",
                               outputCommands = cms.untracked.vstring('drop *',
                                                                      'keep *_ecalUncalib*_*_RECO2',
                                                                      'keep *_ecalRecHit*_*_RECO2',
                                                                      'keep *_offlineBeamSpot_*_*',
                                                                      'keep *_addPileupInfo_*_*'
                                                                      ),
                               fileName = cms.untracked.string('testEcalLocalRecoA.root')
                               )


process.ecalAmplitudeReco = cms.Sequence(process.ecalWeightsUncalibRecHit *
                                         process.ecalMultiFitUncalibRecHit)

process.ecalRecHitsReco = cms.Sequence(process.ecalRecHitWeights
                                       *process.ecalRecHitMultiFit)

process.ecalTestRecoLocal = cms.Sequence(process.raw2digi_step
                                         *process.ecalAmplitudeReco
                                         *process.ecalRecHitsReco
                                         )

from PhysicsTools.PatAlgos.tools.helpers import *
#if isMC:
#     massSearchReplaceAnyInputTag(process.ecalTestRecoLocal,cms.InputTag("ecalDigis:ebDigis"), cms.InputTag("simEcalDigis:ebDigis"),True)
#     massSearchReplaceAnyInputTag(process.ecalTestRecoLocal,cms.InputTag("ecalDigis:eeDigis"), cms.InputTag("simEcalDigis:eeDigis"),True)
if runOnRAW:
    massSearchReplaceAnyInputTag(process.ecalTestRecoLocal,cms.InputTag("ecalDigis:ebDigis"), cms.InputTag("ecalEBunpacker:ebDigis"),True)
    massSearchReplaceAnyInputTag(process.ecalTestRecoLocal,cms.InputTag("ecalDigis:eeDigis"), cms.InputTag("ecalEBunpacker:eeDigis"),True)

process.p = cms.Path(process.ecalTestRecoLocal)
process.outpath = cms.EndPath(process.out)


