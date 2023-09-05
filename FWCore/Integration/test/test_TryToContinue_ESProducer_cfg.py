import FWCore.ParameterSet.Config as cms

import sys
import argparse

parser = argparse.ArgumentParser(prog=sys.argv[0], description='Test TryToContinue exception handling wrt ESProducer.')

parser.add_argument("--continueAnalyzer", help="Apply shouldTryToContinue to module dependent on module that fails.", action="store_true")

argv = sys.argv[:]
if '--' in argv:
    argv.remove("--")
args, unknown = parser.parse_known_args(argv)

process = cms.Process("Demo")

process.MessageLogger = cms.Service("MessageLogger")

process.maxEvents.input = 2

if args.continueAnalyzer:
    process.options.TryToContinue = ['StdException', 'MakeDataException']
    process.options.modulesToCallForTryToContinue = ['demo']
else:
    process.options.TryToContinue = ['StdException']


process.source = cms.Source("EmptySource",
    numberEventsInRun = cms.untracked.uint32(3)
)

#stuck something into the EventSetup
process.WhatsItESProducer = cms.ESProducer("WhatsItESProducer")
process.DoodadESProducer = cms.ESProducer("DoodadESProducer",
    throwException = cms.untracked.bool(True)
)

process.demo = cms.EDAnalyzer("WhatsItAnalyzer",
    expectedValues = cms.untracked.vint32(0)
)

process.bad = cms.ESSource("EmptyESSource",
    recordName = cms.string('GadgetRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

process.p = cms.Path(process.demo)
