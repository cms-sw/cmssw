# Abstract all early deletion settings here

import collections

import FWCore.ParameterSet.Config as cms

from RecoTracker.Configuration.customiseEarlyDeleteForSeeding import customiseEarlyDeleteForSeeding
from CommonTools.ParticleFlow.Isolation.customiseEarlyDeleteForCandIsoDeposits import customiseEarlyDeleteForCandIsoDeposits

def _hasInputTagModuleLabel(process, pset, moduleLabels,result):
    for name in pset.parameterNames_():
        value = getattr(pset,name)
        if isinstance(value, cms.PSet):
            _hasInputTagModuleLabel(process, value, moduleLabels,result)
        elif isinstance(value, cms.VPSet):
            for ps in value:
                _hasInputTagModuleLabel(process, ps, moduleLabels,result)
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
            _hasInputTagModuleLabel(process, getattr(process, value.value()), moduleLabels,result)


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
    for branches in products.itervalues():
        for branch in branches:
            branchSet.add(branch)
    process.options.canDeleteEarly.extend(list(branchSet))

    # LogErrorHarvester should not wait for deleted items
    for prod in process.producers_().itervalues():
        if prod.type_() == "LogErrorHarvester":
            if not hasattr(prod,'excludeModules'):
                prod.excludeModules = cms.untracked.vstring()
            t = prod.excludeModules.value()
            t.extend([b.split('_')[1] for b in branchSet])
            prod.excludeModules = t

    # Find the consumers
    producers=[]
    branchesList=[]
    for producer, branches in products.iteritems():
        producers.append(producer)
        branchesList.append(branches)

    for moduleType in [process.producers_(), process.filters_(), process.analyzers_()]:
        for name, module in moduleType.iteritems():
            result=[]
            for producer in producers:
                result.append(False)

            _hasInputTagModuleLabel(process, module, producers,result)
            for i in range(len(result)):
                if result[i]:
                    if hasattr(module, "mightGet"):
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

            result=[False,False,False,False,False,False,False,False,False,False,False,False,False,False]
            _hasInputTagModuleLabel(p, p.prod, ["foo","foo2","foo3","bar","fred","wilma","a","foo4","bar2","bar3","fred2","wilma2","a2","joe"],result)
            for i in range (0,13):
                self.assert_(result[i])
            self.assert_(not result[13])

    unittest.main()
