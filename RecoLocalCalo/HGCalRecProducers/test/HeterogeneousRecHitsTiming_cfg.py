import os, sys, glob
import FWCore.ParameterSet.Config as cms
from Configuration.StandardSequences.Eras import eras
from Configuration.ProcessModifiers.gpu_cff import gpu
from RecoLocalCalo.HGCalRecProducers.HGCalRecHit_cfi import HGCalRecHit

PU=0
withGPU=0

#package loading
process = cms.Process("gpuTiming", gpu) if withGPU else cms.Process("cpuTiming")
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load('Configuration.Geometry.GeometryExtended2026D49Reco_cff')
process.load('HeterogeneousCore.CUDAServices.CUDAService_cfi')
process.load('RecoLocalCalo.HGCalRecProducers.HGCalRecHit_cfi')
process.load('SimCalorimetry.HGCalSimProducers.hgcalDigitizer_cfi')
process.load( "HLTrigger.Timer.FastTimerService_cfi" )

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic', '')

indir =  '/home/bfontana/' #'/eos/user/b/bfontana/Samples/'
filename_suff = 'hadd_out_PU' + str(PU) + '_uncompressed' #'step3_ttbar_PU' + str(PU)
fNames = [ 'file:' + x for x in glob.glob(os.path.join(indir, filename_suff + '*.root')) ]
if len(fNames)==0:
    print('Used globbing: ', glob.glob(os.path.join(indir, filename_suff + '*.root')))
    raise ValueError('No input files!')
print('Input: ', fNames)
keep = 'keep *'
process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(fNames),
                            inputCommands = cms.untracked.vstring(keep),
                            duplicateCheckMode = cms.untracked.string("noDuplicateCheck") )

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )
wantSummaryFlag = True
process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool( wantSummaryFlag )) #add option for edmStreams

process.load('RecoLocalCalo.HGCalRecProducers.HeterogeneousEERecHitGPU_cfi')
process.load('RecoLocalCalo.HGCalRecProducers.HeterogeneousEERecHitGPUtoSoA_cfi')
process.load('RecoLocalCalo.HGCalRecProducers.HeterogeneousEERecHitFromSoA_cfi')

process.load('RecoLocalCalo.HGCalRecProducers.HeterogeneousHEFRecHitGPU_cfi')
process.load('RecoLocalCalo.HGCalRecProducers.HeterogeneousHEFRecHitGPUtoSoA_cfi')
process.load('RecoLocalCalo.HGCalRecProducers.HeterogeneousHEFRecHitFromSoA_cfi')

process.load('RecoLocalCalo.HGCalRecProducers.HeterogeneousHEBRecHitGPU_cfi')
process.load('RecoLocalCalo.HGCalRecProducers.HeterogeneousHEBRecHitGPUtoSoA_cfi')
process.load('RecoLocalCalo.HGCalRecProducers.HeterogeneousHEBRecHitFromSoA_cfi')

process.ThroughputService = cms.Service( "ThroughputService",
                                         eventRange = cms.untracked.uint32( 300 ),
                                         eventResolution = cms.untracked.uint32( 1 ),
                                         printEventSummary = cms.untracked.bool( wantSummaryFlag ),
                                         enableDQM = cms.untracked.bool( False )
                                         #valid only for enableDQM=True
                                         #dqmPath = cms.untracked.string( "HLT/Throughput" ),
                                         #timeRange = cms.untracked.double( 60000.0 ),
                                         #dqmPathByProcesses = cms.untracked.bool( False ),
                                         #timeResolution = cms.untracked.double( 5.828 )
)

process.FastTimerService.enableDQM = False
process.FastTimerService.writeJSONSummary = True
process.FastTimerService.jsonFileName = 'resources.json'
process.MessageLogger.categories.append('ThroughputService')

if withGPU:
    process.ee_t = cms.Task( process.EERecHitGPUProd, process.EERecHitGPUtoSoAProd, process.EERecHitFromSoAProd )
    process.hef_t = cms.Task( process.HEFRecHitGPUProd, process.HEFRecHitGPUtoSoAProd, process.HEFRecHitFromSoAProd )
    process.heb_t = cms.Task( process.HEBRecHitGPUProd, process.HEBRecHitGPUtoSoAProd, process.HEBRecHitFromSoAProd )
    process.recHitsTask = cms.Task( process.ee_t, process.hef_t, process.heb_t )
    outkeeps = ['keep *_EERecHitFromSoAProd_*_*',
                'keep *_HEFRecHitFromSoAProd_*_*',
                'keep *_HEBRecHitFromSoAProd_*_*']
else:
    process.recHitsClone = HGCalRecHit.clone()
    process.recHitsTask = cms.Task( process.recHitsClone ) #CPU version
    outkeeps = ['keep *_*_' + f + '*_*' for f in ['HGCEERecHits', 'HGCHEFRecHits', 'HGCHEBRecHits'] ]

process.path = cms.Path( process.recHitsTask )

"""
process.consumer = cms.EDAnalyzer("GenericConsumer",                     
                                  eventProducts = cms.untracked.vstring('EERecHitGPUProd',
                                                                        'HEFRecHitGPUProd',
                                                                        'HEBRecHitGPUProd') )
"""
"""
process.consumer = cms.EDAnalyzer('GenericConsumer',
                                  eventProducts = cms.untracked.vstring('recHitsClone') )
                                  #eventProducts = cms.untracked.vstring('HGCalUncalibRecHit') ) #uncalib only (to assess reading speed)
"""
"""
process.consume_step = cms.EndPath(process.consumer)
"""

process.out = cms.OutputModule( "PoolOutputModule", 
                                fileName = cms.untracked.string( '/home/bfontana/out_Timing_PU' + str(PU) + '_' +str(withGPU)+ '.root'),
                                outputCommands = cms.untracked.vstring(outkeeps[0], outkeeps[1], outkeeps[2])
)
process.outpath = cms.EndPath(process.out)


#process.schedule.append(process.consume_step) #in case one has multiple Paths or EndPaths to run
