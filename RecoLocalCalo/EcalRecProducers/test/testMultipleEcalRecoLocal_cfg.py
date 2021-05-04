import FWCore.ParameterSet.Config as cms
process = cms.Process("RECO2")

process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.Geometry.GeometrySimDB_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load('Configuration.StandardSequences.RawToDigi_cff')
process.load('Configuration.StandardSequences.Reconstruction_cff')

process.GlobalTag.globaltag = 'GR_R_74_V10A'
process.GlobalTag.toGet = cms.VPSet(
    cms.PSet(record = cms.string("GeometryFileRcd"),
             tag = cms.string("XMLFILE_Geometry_2015_72YV2_Extended2015ZeroMaterial_mc"),
             connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_GEOMETRY_000"),
#             label = cms.untracked.string("Extended2015ZeroMaterial")
             ),
    cms.PSet(record = cms.string("EcalTBWeightsRcd"),
             tag = cms.string("EcalTBWeights_3p5_time_mc"),
             connect = cms.untracked.string("frontier://FrontierPrep/CMS_COND_ECAL")
             )
    )

#### CONFIGURE IT HERE
isMC = True
#####################
process.MessageLogger.cerr.FwkReport.reportEvery = 1

# start from RAW format for more flexibility
process.raw2digi_step = cms.Sequence(process.RawToDigi)

# get uncalibrechits with global method / time from ratio
import RecoLocalCalo.EcalRecProducers.ecalGlobalUncalibRecHit_cfi
process.ecalGlobalUncalibRecHit = RecoLocalCalo.EcalRecProducers.ecalGlobalUncalibRecHit_cfi.ecalGlobalUncalibRecHit.clone()
# get uncalib rechits from multifit method / time from ratio
import RecoLocalCalo.EcalRecProducers.ecalMultiFitUncalibRecHit_cfi
process.ecalMultiFitUncalibRecHit =  RecoLocalCalo.EcalRecProducers.ecalMultiFitUncalibRecHit_cfi.ecalMultiFitUncalibRecHit.clone()
process.ecalMultiFitUncalibRecHit.algoPSet.activeBXs = cms.vint32(-5,-4,-3,-2,-1,0,1,2,3,4)
process.ecalMultiFitUncalibRecHit.algoPSet.useLumiInfoRunHeader = cms.bool( False )
# get uncalib rechits from multifit method / time from weights
process.ecalMultiFit2UncalibRecHit =  RecoLocalCalo.EcalRecProducers.ecalMultiFitUncalibRecHit_cfi.ecalMultiFitUncalibRecHit.clone()
process.ecalMultiFit2UncalibRecHit.algoPSet.timealgo = cms.string("WeightsMethod")
process.ecalMultiFit2UncalibRecHit.algoPSet.activeBXs = cms.vint32(-5,-4,-3,-2,-1,0,1,2,3,4)
process.ecalMultiFit2UncalibRecHit.algoPSet.useLumiInfoRunHeader = cms.bool ( False )

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


process.ecalRecHitGlobal = process.ecalRecHit.clone()
process.ecalRecHitGlobal.EBuncalibRecHitCollection = 'ecalGlobalUncalibRecHit:EcalUncalibRecHitsEB'
process.ecalRecHitGlobal.EEuncalibRecHitCollection = 'ecalGlobalUncalibRecHit:EcalUncalibRecHitsEE'
process.ecalRecHitGlobal.EBrechitCollection = 'EcalRecHitsGlobalEB'
process.ecalRecHitGlobal.EErechitCollection = 'EcalRecHitsGlobalEE'

process.ecalRecHitMultiFit = process.ecalRecHit.clone()
process.ecalRecHitMultiFit.EBuncalibRecHitCollection = 'ecalMultiFitUncalibRecHit:EcalUncalibRecHitsEB'
process.ecalRecHitMultiFit.EEuncalibRecHitCollection = 'ecalMultiFitUncalibRecHit:EcalUncalibRecHitsEE'
process.ecalRecHitMultiFit.EBrechitCollection = 'EcalRecHitsMultiFitEB'
process.ecalRecHitMultiFit.EErechitCollection = 'EcalRecHitsMultiFitEE'

process.ecalRecHitMultiFit2 = process.ecalRecHit.clone()
process.ecalRecHitMultiFit2.EBuncalibRecHitCollection = 'ecalMultiFit2UncalibRecHit:EcalUncalibRecHitsEB'
process.ecalRecHitMultiFit2.EEuncalibRecHitCollection = 'ecalMultiFit2UncalibRecHit:EcalUncalibRecHitsEE'
process.ecalRecHitMultiFit2.EBrechitCollection = 'EcalRecHitsMultiFit2EB'
process.ecalRecHitMultiFit2.EErechitCollection = 'EcalRecHitsMultiFit2EE'

process.maxEvents = cms.untracked.PSet(  input = cms.untracked.int32(1000) )
path = '/store/data/Run2012D/DoubleElectron/RAW-RECO/ZElectron-22Jan2013-v1/10000/'
process.source = cms.Source("PoolSource",
                            duplicateCheckMode = cms.untracked.string("noDuplicateCheck"),
                            fileNames = cms.untracked.vstring(path+'0008202C-E78F-E211-AADB-0026189437FD.root'
                                                              ))


process.out = cms.OutputModule("PoolOutputModule",
                               outputCommands = cms.untracked.vstring('drop *',
                                                                      'keep *_ecalUncalib*_*_RECO2',
                                                                      'keep *_ecalRecHit*_*_RECO2',
                                                                      'keep *_offlineBeamSpot_*_*',
                                                                      'keep *_addPileupInfo_*_*'
                                                                      ),
                               fileName = cms.untracked.string('reco2_pu40.root')
                               )


process.ecalAmplitudeReco = cms.Sequence( process.ecalGlobalUncalibRecHit *
                                          process.ecalMultiFitUncalibRecHit *
                                          process.ecalMultiFit2UncalibRecHit
                                          )

process.ecalRecHitsReco = cms.Sequence( process.ecalRecHitGlobal *
                                        process.ecalRecHitMultiFit *
                                        process.ecalRecHitMultiFit2 )

process.ecalTestRecoLocal = cms.Sequence( process.raw2digi_step *
                                          process.ecalAmplitudeReco *
                                          process.ecalRecHitsReco )

from PhysicsTools.PatAlgos.tools.helpers import *

process.p = cms.Path(process.ecalTestRecoLocal)
process.outpath = cms.EndPath(process.out)


#########################
#    Time Profiling     #
#########################

#https://twiki.cern.ch/twiki/bin/viewauth/CMS/FastTimerService
process.MessageLogger.cerr.FastReport = cms.untracked.PSet( limit = cms.untracked.int32( 10000000 ) )

# remove any instance of the FastTimerService
if 'FastTimerService' in process.__dict__:
    del process.FastTimerService
    
# instrument the menu with the FastTimerService
process.load( "HLTrigger.Timer.FastTimerService_cfi" )

# print a text summary at the end of the job
process.FastTimerService.printJobSummary          = True

# enable per-event DQM plots
process.FastTimerService.enableDQM                = True

# enable per-module DQM plots
process.FastTimerService.enableDQMbyModule        = True
        
# enable per-event DQM plots by lumisection
process.FastTimerService.enableDQMbyLumiSection   = True
process.FastTimerService.dqmLumiSectionsRange     = 2500    # lumisections (23.31 s)

# set the time resolution of the DQM plots
process.FastTimerService.dqmTimeRange             = 1000.   # ms
process.FastTimerService.dqmTimeResolution        =    5.   # ms
process.FastTimerService.dqmPathTimeRange         =  100.   # ms
process.FastTimerService.dqmPathTimeResolution    =    0.5  # ms
process.FastTimerService.dqmModuleTimeRange       = 1000.   # ms
process.FastTimerService.dqmModuleTimeResolution  =    0.5  # ms

# set the base DQM folder for the plots
process.FastTimerService.dqmPath                  = "HLT/TimerService"
process.FastTimerService.enableDQMbyProcesses     = True

# save the DQM plots in the DQMIO format
process.dqmOutput = cms.OutputModule("DQMRootOutputModule",
                                     fileName = cms.untracked.string("DQM_pu40.root")
                                     )
process.FastTimerOutput = cms.EndPath( process.dqmOutput )
