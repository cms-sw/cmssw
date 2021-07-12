# Abstract all early deletion settings here

import collections

import FWCore.ParameterSet.Config as cms

from RecoTracker.Configuration.customiseEarlyDeleteForSeeding import customiseEarlyDeleteForSeeding
from CommonTools.ParticleFlow.Isolation.customiseEarlyDeleteForCandIsoDeposits import customiseEarlyDeleteForCandIsoDeposits
import six

def _hasInputTagModuleLabel(process, pset, psetModLabel, moduleLabels, result):
    for name in pset.parameterNames_():
        value = getattr(pset,name)
        if isinstance(value, cms.PSet):
            _hasInputTagModuleLabel(process, value, psetModLabel, moduleLabels, result)
        elif isinstance(value, cms.VPSet):
            for ps in value:
                _hasInputTagModuleLabel(process, ps, psetModLabel, moduleLabels, result)
        elif isinstance(value, cms.VInputTag):
            for t in value:
                t2 = t
                if not isinstance(t, cms.InputTag):
                    t2 = cms.InputTag(t2)
                for i,moduleLabel in enumerate(moduleLabels):
                    if result[i]: continue #no need
                    if t2.getModuleLabel() == moduleLabel:
                        result[i]=True
        elif isinstance(value, cms.InputTag):
            for i,moduleLabel in enumerate(moduleLabels):
                if result[i]: continue #no need
                if value.getModuleLabel() == moduleLabel:
                    result[i]=True
        elif isinstance(value, cms.string) and name == "refToPSet_":
            try:
                ps = getattr(process, value.value())
            except AttributeError:
                raise RuntimeError("Module %s has a 'PSet(refToPSet_ = cms.string(\"%s\"))', but the referenced-to PSet does not exist in the Process." % (psetModLabel, value.value()))
            _hasInputTagModuleLabel(process, ps, psetModLabel, moduleLabels, result)


def customiseEarlyDelete(process):
    # Mapping label -> [branches]
    # for the producers whose products are to be deleted early
    products = collections.defaultdict(list)

    products = customiseEarlyDeleteForSeeding(process, products)

    products = customiseEarlyDeleteForCandIsoDeposits(process, products)

    # Set process.options.canDeleteEarly
    if not hasattr(process.options, "canDeleteEarly"):
        process.options.canDeleteEarly = cms.untracked.vstring()

    branchSet = set()
    for branches in six.itervalues(products):
        for branch in branches:
            branchSet.add(branch)
    branchList = sorted(branchSet)
    process.options.canDeleteEarly.extend(branchList)

    # LogErrorHarvester should not wait for deleted items
    for prod in six.itervalues(process.producers_()):
        if prod.type_() == "LogErrorHarvester":
            if not hasattr(prod,'excludeModules'):
                prod.excludeModules = cms.untracked.vstring()
            t = prod.excludeModules.value()
            t.extend([b.split('_')[1] for b in branchList])
            prod.excludeModules = t

    # Find the consumers
    producers=[]
    branchesList=[]
    for producer, branches in six.iteritems(products):
        producers.append(producer)
        branchesList.append(branches)

    for moduleType in [process.producers_(), process.filters_(), process.analyzers_()]:
        for name, module in six.iteritems(moduleType):
            result=[]
            for producer in producers:
                result.append(False)

            _hasInputTagModuleLabel(process, module, name, producers, result)
            for i in range(len(result)):
                if result[i]:
                    #if it exists it might be optional or empty, both evaluate to False
                    if hasattr(module, "mightGet") and module.mightGet:
                        module.mightGet.extend(branchesList[i])
                    else:
                        module.mightGet = cms.untracked.vstring(branchesList[i])
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
            p.prod2 = cms.EDProducer("Producer2",
                foo = cms.PSet(
                    refToPSet_ = cms.string("nonexistent")
                )
            )

            result=[False,False,False,False,False,False,False,False,False,False,False,False,False,False]
            _hasInputTagModuleLabel(p, p.prod, "prod", ["foo","foo2","foo3","bar","fred","wilma","a","foo4","bar2","bar3","fred2","wilma2","a2","joe"], result)
            for i in range (0,13):
                self.assert_(result[i])
            self.assert_(not result[13])

            result = [False]
            self.assertRaises(RuntimeError, _hasInputTagModuleLabel, p, p.prod2, "prod2", ["foo"], result)

    unittest.main()
