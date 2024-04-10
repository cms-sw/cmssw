import FWCore.ParameterSet.Config as cms
import sys

process = cms.Process("Test")


process.source = cms.Source("EmptySource")


mod = int(sys.argv[1])

process.load('FWCore.MessageService.MessageLogger_cfi')
process.MessageLogger.cerr.enable = False
process.MessageLogger.files.exceptionMessage = dict(filename="test_bad_schedule_{}".format(mod),
                                                    noTimeStamps= True)

if mod == 0 :
    #directly depends on
    process.a = cms.EDProducer("AddIntsProducer",
                           labels=cms.VInputTag(cms.InputTag("a")))
    process.p = cms.Path(process.a)
elif mod == 1:
    #cross path dependency
    process.a = cms.EDProducer("AddIntsProducer",
                           labels=cms.VInputTag(cms.InputTag("b")))

    process.b = cms.EDProducer("AddIntsProducer",
                           labels=cms.VInputTag(cms.InputTag("a")))

    process.p = cms.Path(process.a)
    process.p2 = cms.Path(process.b)

elif mod == 2:
    #path ordering
    process.a = cms.EDProducer("AddIntsProducer",
                           labels=cms.VInputTag(cms.InputTag("b")))

    process.b = cms.EDProducer("AddIntsProducer",
                           labels=cms.VInputTag(cms.InputTag("")))
    process.p = cms.Path(process.a + process.b)

elif mod == 3:
    #path ordering, extra path 1
    process.a = cms.EDProducer("AddIntsProducer",
                           labels=cms.VInputTag(cms.InputTag("b")))

    process.b = cms.EDProducer("AddIntsProducer",
                           labels=cms.VInputTag(cms.InputTag("")))
    process.p = cms.Path(process.a + process.b)
    process.p2 = cms.Path(process.a)

elif mod == 4:
    #path ordering, extra path 1
    process.a = cms.EDProducer("AddIntsProducer",
                           labels=cms.VInputTag(cms.InputTag("b")))

    process.b = cms.EDProducer("AddIntsProducer",
                           labels=cms.VInputTag(cms.InputTag("")))
    process.p2 = cms.Path(process.a + process.b)
    process.p = cms.Path(process.a)

elif mod == 5:
    #cycle with unscheduled
    process.a = cms.EDProducer("AddIntsProducer",
                           labels=cms.VInputTag(cms.InputTag("b")))

    process.b = cms.EDProducer("AddIntsProducer",
                           labels=cms.VInputTag(cms.InputTag("a")))

    process.p = cms.Path(process.a, cms.Task(process.b))

elif mod == 6:
    #multi-path cycle with unscheduled
    process.a = cms.EDProducer("AddIntsProducer",
                           labels=cms.VInputTag(cms.InputTag("b")))

    process.b = cms.EDProducer("AddIntsProducer",
                           labels=cms.VInputTag(cms.InputTag("c")))

    process.c = cms.EDProducer("AddIntsProducer",
                           labels=cms.VInputTag(cms.InputTag("d")))
    process.d = cms.EDProducer("AddIntsProducer",
                           labels=cms.VInputTag(cms.InputTag("a")))

    process.p1 = cms.Path(process.a, cms.Task(process.b))
    process.p2 = cms.Path(process.c, cms.Task(process.d))

elif mod == 7:
    #cycle with unscheduled
    process.a = cms.EDProducer("AddIntsProducer",
                           labels=cms.VInputTag(cms.InputTag("b")))

    process.b = cms.EDProducer("AddIntsProducer",
                           labels=cms.VInputTag(cms.InputTag("c")))

    process.c = cms.EDProducer("AddIntsProducer",
                           labels=cms.VInputTag(cms.InputTag("b")))

    process.p1 = cms.Path(process.a, cms.Task(process.b, process.c))

elif mod == 8:
    #cycle with filter on other path
    process.a = cms.EDProducer("IntProducer", ivalue = cms.int32(10))
    process.b = cms.EDProducer("AddIntsProducer", labels = cms.VInputTag("c"))
    process.c = process.b.clone(labels=["a"])
    process.dependingFilter = cms.EDFilter("IntProductFilter", label=cms.InputTag("b"), threshold=cms.int32(1000))
    process.rejectingFilter = cms.EDFilter("TestFilterModule", acceptValue = cms.untracked.int32(-1))

    process.p1 = cms.Path(process.c)
    process.p2 = cms.Path(process.b + process.dependingFilter + process.a)
    process.p3 = cms.Path(process.rejectingFilter + process.a)
