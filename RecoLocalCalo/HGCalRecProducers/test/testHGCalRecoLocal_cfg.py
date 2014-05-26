import FWCore.ParameterSet.Config as cms

process = cms.Process("testHGCalRecoLocal")

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.Geometry.GeometryExtended2023HGCalMuonReco_cff')
process.load('Configuration.Geometry.GeometryExtended2023HGCalMuon_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')

# get timing service up for profiling
process.TimerService = cms.Service("TimerService")
process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
)


# unpacking


# get uncalibrechits with weights method
import RecoLocalCalo.HGCalRecProducers.HGCalUncalibRecHit_cfi
process.HGCalUncalibHit = RecoLocalCalo.HGCalRecProducers.HGCalUncalibRecHit_cfi.HGCalUncalibRecHit.clone()
process.HGCalUncalibHit.HGCEEdigiCollection = 'hgcDigitizer:eeDigis'
process.HGCalUncalibHit.HGCHEFdigiCollection = 'HGCDigitizer:hefDigis'
process.HGCalUncalibHit.HGCHEBdigiCollection = 'HGCDigitizer:hebDigis'

# get rechits e.g. from the weights
process.load("RecoLocalCalo.HGCalRecProducers.HGCalRecHit_cfi")
process.HGCalRecHit.HGCEEuncalibRecHitCollection = 'HGCalUncalibHit:HGCalUncalibRecHitsEE'
process.HGCalRecHit.HGCHEFuncalibRecHitCollection = 'HGCalUncalibHit:HGCalUncalibRecHitsHEF'
process.HGCalRecHit.HGCHEBuncalibRecHitCollection = 'HGCalUncalibHit:HGCalUncalibRecHitsHEB'


process.maxEvents = cms.untracked.PSet(  input = cms.untracked.int32(10) )
process.source = cms.Source("PoolSource",
            fileNames = cms.untracked.vstring('file:step2.root')
                )


process.OutputModule_step = cms.OutputModule("PoolOutputModule",
                                          outputCommands = cms.untracked.vstring(
            'drop *',
            'keep *_HGCalUncalibHit*_*_*',
            'keep *_HGCalRecHit_*_*'
          ),
  fileName = cms.untracked.string('testHGCalLocalRecoA.root')
)

process.dumpEv = cms.EDAnalyzer("EventContentAnalyzer")


# customisation of the process.

# Automatic addition of the customisation function from SLHCUpgradeSimulations.Configuration.combinedCustoms
from SLHCUpgradeSimulations.Configuration.combinedCustoms import cust_2023NoEE 

#call to customisation function cust_2023NoEE imported from SLHCUpgradeSimulations.Configuration.combinedCustoms
process = cust_2023NoEE(process)

# End of customisation functions

process.HGCalRecoLocal = cms.Sequence(process.HGCalUncalibHit
                                         +process.HGCalRecHit
                                         #+process.OutputModule
                                         #+process.dumpEv
                                        )

process.p = cms.Path(process.HGCalRecoLocal)
