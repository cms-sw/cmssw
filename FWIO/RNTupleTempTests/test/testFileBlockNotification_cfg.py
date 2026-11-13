import FWCore.ParameterSet.Config as cms
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--stage", choices=("write", "merge", "read"), required=True, help="Test stage to run")
parser.add_argument("--input", help="Input file for merge or read")
parser.add_argument("--output", help="Output file for write or merge")
options = parser.parse_args()

if bool(options.input) != (options.stage != "write"):
    parser.error(f"--input {'is not used' if options.stage == 'write' else 'is required'} for stage {options.stage}")
if bool(options.output) != (options.stage != "read"):
    parser.error(f"--output {'is not used' if options.stage == 'read' else 'is required'} for stage {options.stage}")

process = cms.Process({"write": "PROD1", "merge": "MERGE", "read": "CHECK"}[options.stage])
process.maxEvents.input = 3 if options.stage == "write" else -1

if options.stage == "write":
    from FWCore.Modules.modules import EmptySource
    process.source = EmptySource()
else:
    from FWIO.RNTupleTempInput.modules import RNTupleTempSource
    process.source = RNTupleTempSource(fileNames=[f"file:{options.input}"])

if options.stage != "read":
    process.pb = cms.EDProducer("IntProducerBeginProcessBlock", ivalue=cms.int32(1))
    process.p = cms.Path(process.pb)

    from FWIO.RNTupleTempOutput.modules import RNTupleTempOutputModule
    process.output = RNTupleTempOutputModule(fileName=options.output)
    process.end = cms.EndPath(process.output)
else:
    from IOPool.Output.modules import edmtest_pool_FileBlockNotificationTest
    process.observer = edmtest_pool_FileBlockNotificationTest(
        nullTrees=True,
        expectedProcesses=[],
        expectedFiles=1,
    )
    process.end = cms.EndPath(process.observer)
