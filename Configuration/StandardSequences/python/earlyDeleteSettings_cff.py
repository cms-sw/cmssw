# Abstract all early deletion settings here

import collections

import FWCore.ParameterSet.Config as cms

from RecoTracker.Configuration.customiseEarlyDeleteForSeeding import customiseEarlyDeleteForSeeding

def _hasInputTagModuleLabel(process, pset, moduleLabel):
    for name in pset.parameterNames_():
        value = getattr(pset,name)
        if isinstance(value, cms.PSet):
            if _hasInputTagModuleLabel(process, value, moduleLabel):
                return True
        elif isinstance(value, cms.VPSet):
            for ps in value:
                if _hasInputTagModuleLabel(process, ps, moduleLabel):
                    return True
        elif isinstance(value, cms.VInputTag):
            for t in value:
                t2 = t
                if not isinstance(t, cms.InputTag):
                    t2 = cms.InputTag(t2)
                if t2.getModuleLabel() == moduleLabel:
                    return True
        elif isinstance(value, cms.InputTag):
            if value.getModuleLabel() == moduleLabel:
                return True
        if isinstance(value, cms.string) and name == "refToPSet_":
            return _hasInputTagModuleLabel(process, getattr(process, value.value()), moduleLabel)
    return False

def customiseEarlyDelete(process):
    # Mapping label -> [branches]
    # for the producers whose products are to be deleted early
    products = collections.defaultdict(list)

    products = customiseEarlyDeleteForSeeding(process, products)

    # Set process.options.canDeleteEarly
    if not hasattr(process.options, "canDeleteEarly"):
        process.options.canDeleteEarly = cms.untracked.vstring()

    branchSet = set()
    for branches in products.itervalues():
        for branch in branches:
            branchSet.add(branch)
    process.options.canDeleteEarly.extend(list(branchSet))

    # Find the consumers
    for moduleType in [process.producers_(), process.filters_(), process.analyzers_()]:
        for name, module in moduleType.iteritems():
            for producer, branches in products.iteritems():
                if _hasInputTagModuleLabel(process, module, producer):
                    #print "Module %s mightGet %s" % (name, str(branches))
                    if hasattr(module, "mightGet"):
                        module.mightGet.extend(branches)
                    else:
                        module.mightGet = cms.untracked.vstring(branches)
    return process


if __name__=="__main__":
    import unittest

    class TestHasInputTagModuleLabel(unittest.TestCase):
        def setUp(self):
            """Nothing to do """
            None
        def testHasInputTagModuleLabel(self):
            p = cms.Process("A")
            p.pset = cms.PSet(a=cms.InputTag("a"),a2=cms.untracked.InputTag("a2"))
            p.prod = cms.EDProducer("Producer",
                foo = cms.InputTag("foo"),
                foo2 = cms.InputTag("foo2", "instance"),
                foo3 = cms.InputTag("foo3", "instance", "PROCESS"),
                foo4 = cms.untracked.InputTag("foo4"),
                nested = cms.PSet(
                    bar = cms.InputTag("bar"),
                    bar2 = cms.untracked.InputTag("bar2"),
                ),
                nested2 = cms.untracked.PSet(
                    bar3 = cms.untracked.InputTag("bar3"),
                ),
                flintstones = cms.VPSet(
                    cms.PSet(fred=cms.InputTag("fred")),
                    cms.PSet(wilma=cms.InputTag("wilma"))
                ),
                flintstones2 = cms.VPSet(
                    cms.PSet(fred2=cms.untracked.InputTag("fred2")),
                    cms.PSet(wilma2=cms.InputTag("wilma2"))
                ),
                ref = cms.PSet(
                    refToPSet_ = cms.string("pset")
                ),
                ref2 = cms.untracked.PSet(
                    refToPSet_ = cms.string("pset")
                ),
            )

            self.assert_(_hasInputTagModuleLabel(p, p.prod, "foo"))
            self.assert_(_hasInputTagModuleLabel(p, p.prod, "foo2"))
            self.assert_(_hasInputTagModuleLabel(p, p.prod, "foo3"))
            self.assert_(_hasInputTagModuleLabel(p, p.prod, "bar"))
            self.assert_(_hasInputTagModuleLabel(p, p.prod, "fred"))
            self.assert_(_hasInputTagModuleLabel(p, p.prod, "wilma"))
            self.assert_(_hasInputTagModuleLabel(p, p.prod, "a"))
            self.assert_(_hasInputTagModuleLabel(p, p.prod, "foo4"))
            self.assert_(_hasInputTagModuleLabel(p, p.prod, "bar2"))
            self.assert_(_hasInputTagModuleLabel(p, p.prod, "bar3"))
            self.assert_(_hasInputTagModuleLabel(p, p.prod, "fred2"))
            self.assert_(_hasInputTagModuleLabel(p, p.prod, "wilma2"))
            self.assert_(_hasInputTagModuleLabel(p, p.prod, "a2"))
            self.assert_(not _hasInputTagModuleLabel(p, p.prod, "joe"))

    unittest.main()
