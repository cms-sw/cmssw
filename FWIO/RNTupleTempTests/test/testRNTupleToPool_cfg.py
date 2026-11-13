import argparse
import FWCore.ParameterSet.Config as cms

parser = argparse.ArgumentParser()
parser.add_argument("--step", required=True, choices=("rntuple", "emptyPool", "convert", "copy", "read", "drop"),
                    help="Test step to run")
parser.add_argument("--input", help="Input file for conversion, copy, or readback")
parser.add_argument("--output", help="Output file for writing or conversion")
args = parser.parse_args()

needs_input = args.step in ("convert", "copy", "read", "drop")
needs_output = args.step in ("rntuple", "emptyPool", "convert", "copy")
if bool(args.input) != needs_input:
    parser.error(f"--input {'is required' if needs_input else 'is not used'} for step {args.step}")
if bool(args.output) != needs_output:
    parser.error(f"--output {'is required' if needs_output else 'is not used'} for step {args.step}")

names = {"rntuple": "PROD", "emptyPool": "PROD", "convert": "CONVERT", "copy": "COPY", "read": "READ", "drop": "DROP"}
process = cms.Process(names[args.step])
process.options = cms.untracked.PSet(numberOfThreads=cms.untracked.uint32(1))

from IOPool.Output.modules import PoolOutputModule

if args.step in ("rntuple", "emptyPool"):
    from FWCore.Modules.modules import EmptySource
    process.source = EmptySource()
    process.maxEvents.input = 3

    from FWCore.Framework.modules import AddIntsProducer, IntProducer
    process.parent = IntProducer(ivalue=7)
    process.middle = AddIntsProducer(labels=["parent"])
    process.child = AddIntsProducer(labels=["middle"])
    process.begin = cms.EDProducer("IntProducerBeginProcessBlock", ivalue=cms.int32(11))
    process.end = cms.EDProducer("IntProducerEndProcessBlock", ivalue=cms.int32(13))
    process.p = cms.Path(process.parent + process.middle + process.child + process.begin + process.end)

    if args.step == "rntuple":
        from FWIO.RNTupleTempOutput.modules import RNTupleTempOutputModule
        process.out = RNTupleTempOutputModule(fileName=args.output)
    else:
        process.out = PoolOutputModule(fileName=args.output)
    process.e = cms.EndPath(process.out)
else:
    if args.step == "convert":
        from FWIO.RNTupleTempInput.modules import RNTupleTempSource
        process.source = RNTupleTempSource(fileNames=["file:" + args.input])
    else:
        from IOPool.Input.modules import PoolSource
        process.source = PoolSource(fileNames=["file:" + args.input])

    from FWCore.Integration.modules import TestFindProduct
    if args.step == "drop":
        process.source.inputCommands = ["keep *", "drop *_parent_*_PROD"]
        process.source.dropDescendantsOfDroppedBranches = True

        process.check = TestFindProduct(
            inputTagsNotFound=["parent", "middle", "child"],
        )
    else:
        process.check = TestFindProduct(
            inputTags=["child::PROD"],
            expectedSum=21,
        )
        from FWCore.Integration.modules import TestParentage
        process.ancestry = TestParentage(
            inputTag="child::PROD",
            expectedAncestors=["parent", "middle"],
        )

    process.checkBegin = TestFindProduct(
        inputTagsInputProcessBlock=["begin::PROD"],
        expectedSum=11,
        expectedCache=11,
    )
    process.checkEnd = TestFindProduct(
        inputTagsInputProcessBlock=["end::PROD"],
        expectedSum=13,
        expectedCache=13,
    )
    process.p = cms.Path(process.check + process.checkBegin + process.checkEnd)

    if args.step != "drop":
        process.p += process.ancestry

        # Each analyzer sees exactly one event, so its expected sum checks that event's value.
        from FWCore.Modules.modules import EventIDFilter
        for event in range(1, 4):
            eventFilter = EventIDFilter(eventsToPass=[f"1:1:{event}"])
            eventCheck = TestFindProduct(
                inputTags=["child::PROD"],
                expectedSum=7,
            )
            setattr(process, f"eventFilter{event}", eventFilter)
            setattr(process, f"eventCheck{event}", eventCheck)
            setattr(process, f"eventPath{event}", cms.Path(eventFilter + eventCheck))

    if args.step in ("convert", "copy"):
        process.out = PoolOutputModule(
            fileName=args.output,
            fastCloning=True,
        )
        process.e = cms.EndPath(process.out)
