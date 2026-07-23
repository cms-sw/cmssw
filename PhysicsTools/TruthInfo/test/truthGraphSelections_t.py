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
            "TTto2L2Nu_Powheg_Pythia8_cfi": ("top", [6, -6]),
            "ST_tch_top_14TeV_TuneCP5_cfi": ("singletop", [6, -6]),
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

    def test_top_and_singletop_keep_production_siblings(self):
        # Single top routes to its own 'singletop' preset and keeps the recoiling
        # W / spectator quark (t+W, t+q); ttbar keeps its associated production
        # system on 'top'. Both seed the top quark.
        self.assertEqual(tgs.templateForFragment("ST_tch_top")[0], "singletop")
        self.assertEqual(tgs.templateForFragment("TTbar_14TeV")[0], "top")
        self.assertEqual(tgs.selectionForFragment("ST_tch_top")["seedPdgIds"], [6, -6])
        self.assertTrue(tgs.selectionForFragment("ST_tch_top")["keepProductionSiblings"])
        self.assertTrue(tgs.selectionForFragment("TTbar_14TeV")["keepProductionSiblings"])

    def test_ttX_routes_to_top(self):
        # ttbar+X (ttH, ttW, ttZ, ttbb, four-top, ttDM) all carry top quarks and
        # seed both tops, like ttbar. SUSY models with a trailing 'tttt' must NOT
        # be caught by the broadened top rule.
        for frag in ["TTH", "TTHH_SL_LO_includeTau", "ttZJets", "TTbb_4f", "TTW", "TTTT_TuneCP5", "ttDM_fragment"]:
            self.assertEqual(tgs.templateForFragment(frag)[0], "top", frag)
            self.assertEqual(tgs.selectionForFragment(frag)["seedPdgIds"], [6, -6], frag)
        self.assertEqual(tgs.templateForFragment("SMS-T1tttt")[0], "full")

    def test_associated_higgs_vh(self):
        # WH / ZH / VH / WWH / ZZH (and HZJ/HWJ orderings): seed the Higgs and keep
        # the recoiling vector boson as a production sibling.
        for frag in ["WH_HToBB_WToLNu", "ZH_HToBB_ZToLL", "WWH_3l", "ZZH", "HZJ_Hee_CT10_13TeV"]:
            self.assertEqual(tgs.templateForFragment(frag)[0], "vh", frag)
            self.assertEqual(tgs.selectionForFragment(frag)["seedPdgIds"], [25], frag)
            self.assertTrue(tgs.selectionForFragment(frag)["keepProductionSiblings"], frag)

    def test_di_higgs_uses_ggf(self):
        # gg -> HH and HH -> ...: seedPdgIds=25 seeds every Higgs (the ggf preset).
        for frag in ["GluGluToHHTo2B2Tau", "HHto2B2G", "HHToWWZZ_4lplus"]:
            self.assertEqual(tgs.templateForFragment(frag)[0], "ggf", frag)
            self.assertEqual(tgs.selectionForFragment(frag)["seedPdgIds"], [25], frag)

    def test_diboson(self):
        for frag in ["WWTo2L2Nu", "WZTo3LNu", "ZZTo4L", "VBS_OSWW_LL_noTop", "WWJJ_SS", "VVTo2L2Nu"]:
            self.assertEqual(tgs.templateForFragment(frag)[0], "diboson", frag)
        self.assertEqual(tgs.selectionForFragment("WZTo3LNu")["seedPdgIds"], [23, 24, -24])
        self.assertTrue(tgs.selectionForFragment("VBS_OSWW")["keepProductionSiblings"])

    def test_wjets_and_dy_njet(self):
        # W+jets and W single-boson seed the W; DrellYan incl. n-jet / dyellell seed the Z.
        for frag in ["WJetsToLNu_TuneCP5", "W4JToLNu", "Wj_enuj_CT10_13TeV", "WtoTauNu_Bin-M-200"]:
            self.assertEqual(tgs.templateForFragment(frag)[0], "resonance", frag)
            self.assertEqual(tgs.selectionForFragment(frag)["seedPdgIds"], [24, -24], frag)
        for frag in ["DY1jToLL_M-50", "dyellell012j_5f_NLO_FXFX", "DYToLL_M-50_14TeV"]:
            self.assertEqual(tgs.templateForFragment(frag)[0], "resonance", frag)
            self.assertEqual(tgs.selectionForFragment(frag)["seedPdgIds"], [23, 32], frag)

    def test_single_top_s_channel(self):
        self.assertEqual(tgs.templateForFragment("ST_s-channel_4f")[0], "singletop")
        self.assertEqual(tgs.selectionForFragment("ST_s-channel_4f")["seedPdgIds"], [6, -6])

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

    def test_pileup_filter_defaults_off_and_overridable(self):
        # Presets never touch pile-up: every archetype keeps all bunch crossings.
        for frag in ["VBFHZZ4Nu", "ZMM_14", "TTbar_14TeV", "SingleMuPt10", "QCD_Pt_80_120"]:
            s = tgs.selectionForFragment(frag)
            self.assertFalse(s["signalOnly"], frag)
            self.assertEqual(s["keepBunchCrossings"], [], frag)
        # The orthogonal filter composes with any preset via overrides.
        s = tgs.selectionForFragment("ZMM_14", signalOnly=True, keepBunchCrossings=[0])
        self.assertTrue(s["signalOnly"])
        self.assertEqual(s["keepBunchCrossings"], [0])
        self.assertIn("--signal-only", tgs.dumperArgs("ZMM_14", signalOnly=True))

    def test_pset_builds(self):
        pset = tgs.postProcessingPSet("VBFHZZ4Nu")
        self.assertTrue(pset.keepProductionSiblings.value())
        self.assertEqual(list(pset.seedPdgIds), [25])
        self.assertEqual(pset.seedParentDepth.value(), 1)
        self.assertFalse(pset.signalOnly.value())
        self.assertEqual(list(pset.keepBunchCrossings), [])
        # Override flows through to the cms.PSet.
        self.assertTrue(tgs.postProcessingPSet("ZMM_14", signalOnly=True).signalOnly.value())


if __name__ == "__main__":
    unittest.main()
