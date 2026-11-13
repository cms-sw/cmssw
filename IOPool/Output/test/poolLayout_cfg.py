import argparse

import FWCore.ParameterSet.Config as cms

parser = argparse.ArgumentParser()
parser.add_argument('--mode', required=True, choices=['input', 'clone', 'copy', 'merge', 'mismatch', 'override', 'overridefast'],
                    help='Test mode to run')
parser.add_argument('--second', action='store_true', help='Create the second input file')
args = parser.parse_args()

process = cms.Process('LAYOUT' if args.mode == 'input' else 'COPY')
process.options.numberOfStreams = 1
process.maxEvents.input = 6 if args.mode == 'input' else -1

if args.mode == 'input':
    from FWCore.Modules.modules import EmptySource
    process.source = EmptySource(
        firstRun=2 if args.second else 1,
        firstLuminosityBlock=1,
        numberEventsInLuminosityBlock=3,
    )

    from FWCore.Integration.modules import ThingProducer
    from FWCore.Framework.modules import IntProducer
    process.alpha = ThingProducer(nThings=4)
    process.beta = IntProducer(ivalue=42)
    process.block = cms.EDProducer('IntProducerBeginProcessBlock', ivalue=cms.int32(73))
    process.p = cms.Path(process.alpha * process.beta * process.block)
else:
    files = ['file:layout_input.root']
    if args.mode in ('mismatch', 'override', 'overridefast'):
        files.append('file:layout_input2.root')
    from IOPool.Input.modules import PoolSource
    process.source = PoolSource(
        fileNames=files,
        duplicateCheckMode='noDuplicateCheck',
    )

from IOPool.Output.modules import PoolOutputModule
if args.mode == 'input' and args.second:
    outputFileName = 'layout_input2.root'
elif args.mode == 'input':
    outputFileName = 'layout_input.root'
else:
    outputFileName = f'layout_{args.mode}.root'

process.out = PoolOutputModule(
    fileName=outputFileName,
    compressionAlgorithm='ZLIB' if args.mode == 'input' else 'ZSTD',
    compressionLevel=1 if args.mode == 'input' else 5,
    basketSize=24576 if args.second else 16384,
    eventAuxiliaryBasketSize=12288 if args.mode == 'input' else 32768,
    compactEventAuxiliary=False,
    fastCloning=args.mode not in ('copy', 'merge', 'override'),
    mergeJob=args.mode == 'merge',
    overrideInputFileSplitLevels=args.mode in ('override', 'overridefast'),
)
process.end = cms.EndPath(process.out)
