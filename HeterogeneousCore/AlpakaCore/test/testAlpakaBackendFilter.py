import os
import sys

# choose a different alpaka backend depending on the SCRAM test being run
try:
    backend = os.environ['SCRAM_ALPAKA_BACKEND']
except:
    backend = 'SerialSync'

# map the alpaka backends to the process accelerators
accelerators = {
    'SerialSync': 'cpu',
    'CudaAsync':  'gpu-nvidia',
    'ROCmAsync':  'gpu-amd'
}

print(f"Testing the alpaka backend {backend} using the process accelerator {accelerators[backend]}")

import FWCore.ParameterSet.Config as cms
from HeterogeneousCore.AlpakaCore.functions import *

process = cms.Process('Test')

process.options.accelerators = [ accelerators[backend] ]

process.maxEvents.input = 10

process.source = cms.Source('EmptySource')

process.load('Configuration.StandardSequences.Accelerators_cff')
process.load('HeterogeneousCore.AlpakaCore.ProcessAcceleratorAlpaka_cfi')

process.alpakaBackendProducer = cms.EDProducer('AlpakaBackendProducer@alpaka')

process.alpakaBackendFilter = cms.EDFilter('AlpakaBackendFilter',
  producer = cms.InputTag('alpakaBackendProducer', 'backend'),
  backends = cms.vstring(backend)
)

process.mustRun = cms.EDProducer("edmtest::MustRunIntProducer", ivalue=cms.int32(1))
process.mustNotRun = cms.EDProducer("FailingProducer")

process.SelectedBackend = cms.Path(process.alpakaBackendProducer + process.alpakaBackendFilter + process.mustRun)
process.AnyOtherBackend = cms.Path(process.alpakaBackendProducer + ~process.alpakaBackendFilter + process.mustNotRun)

process.options.wantSummary = True
