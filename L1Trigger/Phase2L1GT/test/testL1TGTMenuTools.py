#!/usr/bin/env python3
"""Unit tests for L1Trigger.Phase2L1GT.l1tGTMenuTools (structured P2GT menu resolution)."""
import unittest

import FWCore.ParameterSet.Config as cms
from L1Trigger.Phase2L1GT.l1tGTMenuTools import flattenGTMenu, offline_to_online_pt


def ref(name):
    return cms.PSet(refToPSet_=cms.string(name))


def make_process():
    process = cms.Process("TEST")
    process.scaling3 = cms.PSet(regionsOffset=cms.vdouble(1.0, 2.0, 3.0), regionsSlope=cms.vdouble(1.1, 1.0, 1.0))
    process.scaling1 = cms.PSet(regionsOffset=cms.vdouble(10.0), regionsSlope=cms.vdouble(2.0))
    process.identityScaling = cms.PSet(regionsOffset=cms.vdouble(0.0), regionsSlope=cms.vdouble(1.0))
    process.jetScaling = cms.PSet(regionsOffset=cms.vdouble(20.0, 20.0, 20.0), regionsSlope=cms.vdouble(1.0, 1.0, 1.0),
                                  minOnlinePt=cms.double(25))
    process.muon = cms.PSet(
        tag=cms.InputTag("l1tGTProducer", "GMTTkMuons"),
        maxEta=cms.double(2.4),
        regionsAbsEtaLowerBounds=cms.vdouble(0, 0.83, 1.24),
        qualityFlags=cms.uint32(1),
        ptScaling=ref("scaling3"),
    )
    process.muonAlias = ref("muon")  # references can be chained
    process.sum = cms.PSet(tag=cms.InputTag("l1tGTProducer", "CL2HtSum"), ptScaling=ref("scaling1"))
    process.jet = cms.PSet(tag=cms.InputTag("l1tGTProducer", "CL2JetsSC4"),
                           regionsAbsEtaLowerBounds=cms.vdouble(0, 1.5, 2.4), ptScaling=ref("jetScaling"))
    return process


class TestFlatten(unittest.TestCase):
    def test_single_object_merge_and_override(self):
        p = make_process()
        p.cond = cms.EDFilter("L1GTSingleObjectCond", object=ref("muonAlias"), offlineMinPt=cms.double(22),
                              maxEta=cms.double(2.1), primVertTag=cms.InputTag("x"))
        flattenGTMenu(p)
        self.assertEqual(p.cond.tag.getProductInstanceLabel(), "GMTTkMuons")
        self.assertEqual(p.cond.maxEta.value(), 2.1)  # collection level wins
        self.assertEqual(p.cond.qualityFlags.value(), 1)
        self.assertEqual(list(p.cond.regionsMinPt), [round(21 / 1.1, 1), 20.0, 19.0])
        for name in ("object", "ptScaling", "offlineMinPt"):
            self.assertFalse(hasattr(p.cond, name))
        self.assertEqual(p.cond.primVertTag.getModuleLabel(), "x")

    def test_collections_single_region_and_scaling_override(self):
        p = make_process()
        p.cond = cms.EDFilter("L1GTDoubleObjectCond",
                              collection1=cms.PSet(object=ref("sum"), offlineMinScalarSumPt=cms.double(50)),
                              collection2=cms.PSet(object=ref("sum"), offlineMinPt=cms.double(50),
                                                   ptScaling=ref("identityScaling")),
                              maxDz=cms.double(1))
        flattenGTMenu(p)
        self.assertEqual(p.cond.collection1.minScalarSumPt.value(), 20.0)
        self.assertEqual(p.cond.collection2.minPt.value(), 50.0)  # overridden scaling
        self.assertEqual(p.cond.maxDz.value(), 1)

    def test_min_online_pt(self):
        p = make_process()
        p.cond = cms.EDFilter("L1GTDoubleObjectCond",
                              collection1=cms.PSet(object=ref("jet"), offlineMinPt=cms.double(30)),
                              collection2=cms.PSet(object=ref("jet"), offlineMinPt=cms.double(100)))
        flattenGTMenu(p)
        self.assertEqual(list(p.cond.collection1.regionsMinPt), [25.0, 25.0, 25.0])
        self.assertEqual(list(p.cond.collection2.regionsMinPt), [80.0, 80.0, 80.0])

    def test_explicit_online_thresholds_kept(self):
        p = make_process()
        p.cond = cms.EDFilter("L1GTDoubleObjectCond",
                              collection1=cms.PSet(object=ref("muon"), regionsMinPt=cms.vdouble(7, 7, 7)),
                              collection2=cms.PSet(tag=cms.InputTag("a", "b"), minPt=cms.double(3)))
        flattenGTMenu(p)
        self.assertEqual(list(p.cond.collection1.regionsMinPt), [7, 7, 7])
        self.assertEqual(p.cond.collection2.minPt.value(), 3)

    def test_errors(self):
        p = make_process()
        p.cond = cms.EDFilter("L1GTSingleObjectCond", object=ref("muon"), offlineMinPt=cms.double(22),
                              regionsMinPt=cms.vdouble(1, 2, 3))
        self.assertRaises(RuntimeError, flattenGTMenu, p)

        p = make_process()
        p.cond = cms.EDFilter("L1GTSingleObjectCond", tag=cms.InputTag("a", "b"), offlineMinPt=cms.double(22))
        self.assertRaises(RuntimeError, flattenGTMenu, p)

        p = make_process()
        p.cond = cms.EDFilter("L1GTSingleObjectCond", object=ref("doesNotExist"))
        self.assertRaises(RuntimeError, flattenGTMenu, p)

    def test_rounding_like_python(self):
        for thr, off, slope in [(22, 1.8, 1.0), (0.25, 0, 1), (0.35, 0, 1), (8, 10, 1.2), (37, 3.58, 1.17)]:
            self.assertEqual(offline_to_online_pt(thr, off, slope), float(max(0, round((thr - off) / slope, 1))))

    def test_step1_2024_menu(self):
        p = cms.Process("TEST")
        p.load("L1Trigger.Configuration.Phase2GTMenus.SeedDefinitions.step1_2024.l1tGTMenu_cff")
        flattenGTMenu(p)  # raises if anything is left unresolved
        self.assertEqual(list(p.SingleTkMuon22.regionsMinPt), [20.2, 20.3, 20.3])


if __name__ == "__main__":
    unittest.main()
