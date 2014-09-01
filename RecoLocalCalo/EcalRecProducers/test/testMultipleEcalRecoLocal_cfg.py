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
#####################
process.MessageLogger.cerr.FwkReport.reportEvery = 1

# start from RAW format for more flexibility
process.raw2digi_step = cms.Sequence(process.RawToDigi)

# get uncalibrechits with weights method
import RecoLocalCalo.EcalRecProducers.ecalWeightUncalibRecHit_cfi
process.ecalWeightsUncalibRecHit = RecoLocalCalo.EcalRecProducers.ecalWeightUncalibRecHit_cfi.ecalWeightUncalibRecHit.clone()
# get uncalib rechits from multifit method
import RecoLocalCalo.EcalRecProducers.ecalMultiFitUncalibRecHit_cfi
process.ecalMultiFitUncalibRecHit =  RecoLocalCalo.EcalRecProducers.ecalMultiFitUncalibRecHit_cfi.ecalMultiFitUncalibRecHit.clone()

# get the recovered digis
if isMC:
    process.ecalDetIdToBeRecovered.ebSrFlagCollection = 'simEcalDigis:ebSrFlags'
    process.ecalDetIdToBeRecovered.eeSrFlagCollection = 'simEcalDigis:eeSrFlags'
    process.ecalRecHit.recoverEBFE = False
    process.ecalRecHit.recoverEEFE = False
    process.ecalRecHit.killDeadChannels = False
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
                            fileNames = cms.untracked.vstring('/store/group/comm_ecal/localreco/cmssw_720p4/photongun_pu25_ave40/photongun_pu25_ave40_lsf_750.root')) 


process.out = cms.OutputModule("PoolOutputModule",
                               outputCommands = cms.untracked.vstring('drop *',
                                                                      'keep *_ecalUncalib*_*_RECO2',
                                                                      'keep *_ecalRecHit*_*_RECO2',
                                                                      'keep *_offlineBeamSpot_*_*',
                                                                      'keep *_addPileupInfo_*_*'
                                                                      ),
                               fileName = cms.untracked.string('reco2.root')
                               )


process.ecalAmplitudeReco = cms.Sequence( process.ecalWeightsUncalibRecHit *
                                          process.ecalMultiFitUncalibRecHit )

process.ecalRecHitsReco = cms.Sequence( process.ecalRecHitWeights *
                                        process.ecalRecHitMultiFit )

process.ecalTestRecoLocal = cms.Sequence( process.raw2digi_step *
                                          process.ecalAmplitudeReco *
                                          process.ecalRecHitsReco )

from PhysicsTools.PatAlgos.tools.helpers import *

process.p = cms.Path(process.ecalTestRecoLocal)
process.outpath = cms.EndPath(process.out)


