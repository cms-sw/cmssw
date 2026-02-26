import FWCore.ParameterSet.Config as cms

import argparse
import sys

parser = argparse.ArgumentParser(prog=sys.argv[0],
    description='Test of the L1TZDCEtSumsProducer plugin')

parser.add_argument('-f', '--fileNames', dest='fileNames', nargs='+',
    default=['/store/hidata/HIRun2024B/HIForward0/RAW/v1/000/388/784/00000/a277c6d8-c445-4d2b-a45c-0e74e4ed8ce8.root'],
    help='Input EDM file(s)'
)

parser.add_argument('-l', '--rawDataLabel', dest='rawDataLabel', type=str, default='rawDataRepacker',
    help="Label of the FEDRawDataCollection product to be used as input")

parser.add_argument('-g', '--globalTag', dest='globalTag', type=str, default='auto:run3_hlt_relval',
    help="Name of the GlobalTag")

parser.add_argument('-n', '--maxEvents', dest='maxEvents', type=int, default=10,
    help="Max number of events to be processed")

parser.add_argument('--skipEvents', dest='skipEvents', type=int, default=0,
    help="Value of process.source.skipEvents")

parser.add_argument('-t', '--numberOfThreads', dest='numberOfThreads', type=int, default=1,
    help="Value of process.options.numberOfThreads")

parser.add_argument('-s', '--numberOfStreams', dest='numberOfStreams', type=int, default=0,
    help="Value of process.options.numberOfStreams")

args = parser.parse_args()

process = cms.Process('TEST')

process.maxEvents.input = args.maxEvents

process.options.numberOfThreads = args.numberOfThreads
process.options.numberOfStreams = args.numberOfStreams
process.options.wantSummary = False

# MessageLogger
process.load('FWCore.MessageService.MessageLogger_cfi')
process.MessageLogger.cerr.FwkReport.reportEvery = 1

# Input source
process.source = cms.Source('PoolSource',
    fileNames = cms.untracked.vstring(args.fileNames),
    skipEvents = cms.untracked.uint32(args.skipEvents)
)

# GlobalTag (ESSource)
from Configuration.AlCa.GlobalTag import GlobalTag
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.GlobalTag = GlobalTag(process.GlobalTag, args.globalTag)

# EventSetup modules
process.HcalTopologyIdealEP = cms.ESProducer("HcalTopologyIdealEP",
    Exclude = cms.untracked.string( "" ),
    MergePosition = cms.untracked.bool( True ),
    appendToDataLabel = cms.string( "" )
)

process.hcalDDDRecConstants = cms.ESProducer("HcalDDDRecConstantsESModule",
    appendToDataLabel = cms.string( "" )
)

process.hcalDDDSimConstants = cms.ESProducer("HcalDDDSimConstantsESModule",
    appendToDataLabel = cms.string( "" )
)

process.zdcTopologyEP = cms.ESProducer("ZdcTopologyEP",
    appendToDataLabel = cms.string( "" )
)

# EventData modules
from EventFilter.L1TRawToDigi.gtStage2Digis_cfi import gtStage2Digis as _gtStage2Digis
process.gtStage2Digis = _gtStage2Digis.clone(InputLabel = args.rawDataLabel)

from EventFilter.HcalRawToDigi.HcalRawToDigi_cfi import hcalDigis as _hcalDigis
process.hcalDigis = _hcalDigis.clone(InputLabel = args.rawDataLabel)

from L1Trigger.L1TZDC.l1tZDCEtSums_cfi import l1tZDCEtSums as _l1tZDCEtSums
process.l1tZDCEtSums = _l1tZDCEtSums.clone(hcalTPDigis = 'hcalDigis')

from L1Trigger.L1TCalorimeter.l1tEtSumsPrinter_cfi import l1tEtSumsPrinter as _l1tEtSumsPrinter
process.l1tZDCEtSumsPrinter1 = _l1tEtSumsPrinter.clone(etSumTypes = [27, 28], src = 'gtStage2Digis:EtSumZDC')
process.l1tZDCEtSumsPrinter2 = process.l1tZDCEtSumsPrinter1.clone(src = 'l1tZDCEtSums')

# Path
process.ThePath = cms.Path(
    process.gtStage2Digis
  + process.hcalDigis
  + process.l1tZDCEtSums
  + process.l1tZDCEtSumsPrinter1
  + process.l1tZDCEtSumsPrinter2
)
