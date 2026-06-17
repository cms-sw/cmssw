#!/usr/bin/env python3
# Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>
# Part of the MC-truth-graph prototype - under heavy development, not yet open
# to external contributions (see PhysicsTools/TruthInfo/README.md).

"""Unit tests for the per-process truth-graph selection presets."""

import unittest

import PhysicsTools.TruthInfo.truthGraphSelections as tgs


class TestTruthGraphSelections(unittest.TestCase):
    def test_library_fragments(self):
        # Every enableTruth relval-library fragment resolves to the right preset.
        expected = {
            "SingleElectronPt35": ("gun", [11, -11]),
            "TenTau_E_15_500": ("gun", [15, -15]),
            "TTbar_14TeV_TuneCP5_cfi": ("top", [6, -6]),
            "DYToLL_M-50_14TeV": ("resonance", [23, 32]),
            "DYToTauTau_M-50_14TeV": ("resonance", [23, 32]),
            "ZMM_14": ("resonance", [23, 32]),
            "H125GGgluonfusion_14TeV": ("ggf", [25]),
            "VBFHZZ4Nu_14TeV": ("vbf", [25]),
        }
        for frag, (template, seeds) in expected.items():
            self.assertEqual(tgs.templateForFragment(frag)[0], template, frag)
            self.assertEqual(tgs.selectionForFragment(frag)["seedPdgIds"], seeds, frag)

    def test_gun_species(self):
        cases = {
            "SingleMuPt10_Eta2p85": [13, -13],
            "SingleGammaPt35": [22],
            "SinglePiE50HCAL": [211, -211],
            "FourMuPt_1_200": [13, -13],
            "CloseByParticle_Photon": [22],
            "CE_E_Front_300um": [0],  # configurable species -> full graph fallback
        }
        for frag, seeds in cases.items():
            self.assertEqual(tgs.templateForFragment(frag)[0], "gun", frag)
            self.assertEqual(tgs.selectionForFragment(frag)["seedPdgIds"], seeds, frag)

    def test_vbf_keeps_production_siblings(self):
        self.assertTrue(tgs.selectionForFragment("VBFHZZ4Nu")["keepProductionSiblings"])
        self.assertTrue(tgs.selectionForFragment("QQToHToTauTau")["keepProductionSiblings"])
        # ggF gg->H is 2->1: no production-vertex co-products.
        self.assertFalse(tgs.selectionForFragment("H125GGgluonfusion")["keepProductionSiblings"])

    def test_channel_groups(self):
        self.assertEqual(tgs.selectionForFragment("ZMM_14")["decayPdgIdGroups"], [[13, -13]])
        self.assertEqual(tgs.selectionForFragment("ZEE_14")["decayPdgIdGroups"], [[11, -11]])
        self.assertEqual(tgs.selectionForFragment("DYToTauTau")["decayPdgIdGroups"], [[15, -15]])

    def test_heavy_flavor(self):
        self.assertEqual(tgs.selectionForFragment("BsToMuMu")["seedHadronFlavors"], [5])
        self.assertEqual(tgs.selectionForFragment("JpsiMM")["seedHadronFlavors"], [4])
        # Heavy flavor seeds by flavor content, not PDG -> the dumper must not pass -s.
        self.assertNotIn("-s", tgs.dumperArgs("BsToMuMu"))
        self.assertIn("-f", tgs.dumperArgs("BsToMuMu"))

    def test_full_fallback(self):
        for frag in ["QCD_Pt_80_120", "MinBias_14TeV", "SingleNuE10", "SMS-T1tttt", "TotallyUnknownXyz"]:
            self.assertEqual(tgs.templateForFragment(frag)[0], "full", frag)
            self.assertEqual(tgs.selectionForFragment(frag)["seedPdgIds"], [0], frag)

    def test_full_customisation(self):
        # Keyword overrides win over the preset.
        s = tgs.selectionForFragment("VBFHZZ4Nu", seedParentDepth=3, keepProductionSiblings=False)
        self.assertEqual(s["seedParentDepth"], 3)
        self.assertFalse(s["keepProductionSiblings"])
        # An explicit template can be forced regardless of the name.
        self.assertEqual(tgs.selectionForFragment(template="full")["seedPdgIds"], [0])

    def test_pset_builds(self):
        pset = tgs.postProcessingPSet("VBFHZZ4Nu")
        self.assertTrue(pset.keepProductionSiblings.value())
        self.assertEqual(list(pset.seedPdgIds), [25])
        self.assertEqual(pset.seedParentDepth.value(), 1)


if __name__ == "__main__":
    unittest.main()
