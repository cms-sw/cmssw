import argparse

import FWCore.ParameterSet.Config as cms

parser = argparse.ArgumentParser()
parser.add_argument('--mode', required=True, choices=['produce', 'merge', 'noclone', 'read'], help='Test mode to run')
parser.add_argument('--part', type=int, default=1, help='Input file number to produce')
args = parser.parse_args()

from IOPool.Input.modules import PoolSource
from IOPool.Output.modules import PoolOutputModule

if args.mode == 'produce':
    process = cms.Process('INPUT')

    from FWCore.Modules.modules import EmptySource
    process.source = EmptySource(
        firstEvent=2 * args.part - 1,
        firstLuminosityBlock=args.part,
    )
    process.maxEvents.input = 2

    from IOPool.Output.modules import edmtest_pool_TransitionValueProducer
    from FWCore.Framework.modules import AddIntsProducer
    process.parent = edmtest_pool_TransitionValueProducer()
    process.child = AddIntsProducer(labels=['parent'])
    process.specialParent = edmtest_pool_TransitionValueProducer()
    process.special = AddIntsProducer(
        labels=['specialParent'],
        onlyGetOnEvent=3,
    )
    process.block = cms.EDProducer('IntProducerBeginProcessBlock', ivalue=cms.int32(17))
    process.p = cms.Path(process.parent * process.child * process.specialParent * process.special * process.block)

    process.out = PoolOutputModule(
        fileName='file:transitionInput%d.root' % args.part,
    )
    process.e = cms.EndPath(process.out)
elif args.mode in ('merge', 'noclone'):
    process = cms.Process('MERGE')
    process.source = PoolSource(
        fileNames=['file:transitionInput%d.root' % i for i in range(1, 4)],
        duplicateCheckMode='noDuplicateCheck',
    )
    process.maxEvents.input = -1

    process.mOutput = PoolOutputModule(
        fileName='file:transitionOutput%s.root' %
            {'merge': '', 'noclone': 'NoClone'}[args.mode],
        outputCommands=['drop *', 'keep *_parent_*_INPUT', 'keep *_child__INPUT',
                        'keep *_specialParent_*_INPUT', 'keep *_special__INPUT'],
        fastCloning=args.mode != 'noclone',
        compactEventAuxiliary=False,
    )
    process.e = cms.EndPath(process.mOutput)
else:
    process = cms.Process('READ')
    process.source = PoolSource(fileNames=['file:transitionOutput.root'])
    process.maxEvents.input = -1

    from IOPool.Output.modules import edmtest_pool_TransitionValueChecker
    process.check = edmtest_pool_TransitionValueChecker(
        src=('child', '', 'INPUT'),
        parent=('parent', '', 'INPUT'),
        special=('special', '', 'INPUT'),
        specialParent=('specialParent', '', 'INPUT'),
    )
    process.e = cms.EndPath(process.check)
