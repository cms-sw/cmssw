#!/usr/bin/env python3
# Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>
# Part of the MC-truth-graph prototype - under heavy development, not yet open
# to external contributions (see PhysicsTools/TruthInfo/README.md).

"""Render the truth-Branch DQM validation plots, inspired by
Validation/HGCalValidation/scripts/makeHGCalValidationPlots.py but self-contained
(PyROOT only). It reads one or more DQM files - either the analyzer DQMIO output
(per-type TH1Fs/TProfiles trees) or a harvested legacy DQM_V0001 file - locates the
Branch-validator folders, derives the efficiency / fake-rate / merge-rate /
duplicate-rate ratios from their numerator/denominator histograms (with binomial
errors) and overlays the booked quality distributions. Passing several files
overlays the samples in one set of plots (e.g. Tau vs ZMM vs TTbar), which doubles
as the per-event guided comparison. Output: one PNG per plot plus an index.html.

Examples:
  makeTruthGraphValidationPlots.py branch_reco_dqm.root -o plots
  makeTruthGraphValidationPlots.py tau.root:Tau zmm.root:ZMM ttbar.root:TTbar -o plots
"""

import os
import argparse

import ROOT

ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)
ROOT.gErrorIgnoreLevel = ROOT.kWarning

# Each folder: ratio plots (name -> (num, den, y-title)) and direct distributions.
FOLDERS = [
    ("Tracking/BranchValidator/TrackingParticle", "Branch vs TrackingParticle", {
        "ratios": {
            "efficiency_eta": ("effnum_eta", "denom_eta", "Branch reproduces TP"),
            "efficiency_pt": ("effnum_pt", "denom_pt", "Branch reproduces TP"),
        },
        "dists": ["completeness_hits", "shared_hits"],
    }),
    ("Tracking/BranchValidator/recoTrack", "Reco track vs Branch", {
        "ratios": {
            "efficiency_eta": ("effnum_eta", "denom_eta", "efficiency"),
            "efficiency_pt": ("effnum_pt", "denom_pt", "efficiency"),
            "fakerate_eta": ("fakenum_eta", "recodenom_eta", "fake rate"),
            "mergerate_eta": ("mergenum_eta", "recodenom_eta", "merge rate"),
            "duplicate_eta": ("dupnum_eta", "denom_eta", "duplicate rate"),
        },
        "dists": ["purity"],
    }),
    ("HGCAL/BranchValidator/Trackster", "Trackster vs Branch", {
        "ratios": {
            "efficiency_eta": ("effnum_eta", "denom_eta", "efficiency"),
            "efficiency_energy": ("effnum_energy", "denom_energy", "efficiency"),
            "fakerate_eta": ("fakenum_eta", "recodenom_eta", "fake rate"),
            "mergerate_eta": ("mergenum_eta", "recodenom_eta", "merge rate"),
        },
        "dists": ["purity"],
    }),
    ("HGCAL/BranchValidator/CaloParticle", "Branch vs CaloParticle", {
        "ratios": {
            "efficiency_eta": ("effnum_eta", "denom_eta", "Branch reproduces CP"),
            "efficiency_pt": ("effnum_pt", "denom_pt", "Branch reproduces CP"),
            "efficiency_energy": ("effnum_energy", "denom_energy", "Branch reproduces CP"),
        },
        "dists": ["purity", "completeness_hits", "completeness_energy", "energy_response"],
    }),
    ("HGCAL/BranchValidator/SimCluster", "Branch vs SimCluster", {
        "ratios": {
            "efficiency_eta": ("effnum_eta", "denom_eta", "Branch reproduces SC"),
            "efficiency_pt": ("effnum_pt", "denom_pt", "Branch reproduces SC"),
            "efficiency_energy": ("effnum_energy", "denom_energy", "Branch reproduces SC"),
        },
        "dists": ["purity", "completeness_hits", "completeness_energy", "energy_response"],
    }),
]

COLORS = [ROOT.kBlack, ROOT.kRed + 1, ROOT.kAzure + 1, ROOT.kGreen + 2, ROOT.kMagenta + 1, ROOT.kOrange + 7]


class DQMReader:
    """Reads MonitorElements by full path from either DQMIO trees or a legacy file."""

    def __init__(self, path):
        self.file = ROOT.TFile.Open(path)
        if not self.file or self.file.IsZombie():
            raise IOError("cannot open %s" % path)
        self.byPath = {}
        self._index_dqmio()
        self.legacyBase = None
        if not self.byPath:
            self.legacyBase = self._find_legacy_base()

    def _index_dqmio(self):
        for tname in ("TH1Fs", "TH1Ds", "TProfiles"):
            t = self.file.Get(tname)
            if not t:
                continue
            for i in range(t.GetEntries()):
                t.GetEntry(i)
                self.byPath[str(t.FullName)] = (tname, i)

    def _find_legacy_base(self):
        # DQMData/Run <n>/<top>/Run summary/<sub...>
        dqm = self.file.Get("DQMData")
        if not dqm:
            return None
        for k in dqm.GetListOfKeys():
            if k.GetName().startswith("Run "):
                return "DQMData/%s" % k.GetName()
        return None

    def get(self, folder, name):
        full = "%s/%s" % (folder, name)
        if full in self.byPath:
            tname, i = self.byPath[full]
            t = self.file.Get(tname)
            t.GetEntry(i)
            return t.Value.Clone()
        if self.legacyBase:
            top, _, sub = folder.partition("/")
            path = "%s/%s/Run summary/%s/%s" % (self.legacyBase, top, sub, name)
            obj = self.file.Get(path)
            if obj:
                return obj.Clone()
        return None


def ratio_hist(num, den, ytitle):
    h = num.Clone()
    h.Reset()
    h.Divide(num, den, 1.0, 1.0, "B")  # binomial errors
    h.GetYaxis().SetTitle(ytitle)
    h.GetYaxis().SetRangeUser(0.0, 1.15)
    return h


def draw_overlay(hists_labels, title, outpath):
    c = ROOT.TCanvas("c", title, 700, 600)
    leg = ROOT.TLegend(0.62, 0.78, 0.88, 0.90)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    drawn = False
    keep = []
    for idx, (h, label) in enumerate(hists_labels):
        if not h or h.GetEntries() == 0:
            continue
        col = COLORS[idx % len(COLORS)]
        h.SetLineColor(col)
        h.SetMarkerColor(col)
        h.SetMarkerStyle(20)
        h.SetMarkerSize(0.7)
        h.SetTitle(title)
        h.Draw("E1" if not drawn else "E1 SAME")
        leg.AddEntry(h, label, "lep")
        keep.append(h)
        drawn = True
    if not drawn:
        c.Close()
        return False
    leg.Draw()
    c.SaveAs(outpath)
    c.Close()
    return True


def main(opts):
    os.makedirs(opts.out, exist_ok=True)
    samples = []
    for spec in opts.inputs:
        path, _, label = spec.partition(":")
        samples.append((label or os.path.splitext(os.path.basename(path))[0], DQMReader(path)))

    made = []  # (folder_title, png_basename)
    for folder, ftitle, spec in FOLDERS:
        prefix = folder.replace("/", "_")
        # Ratio plots.
        for outname, (num, den, ytitle) in spec["ratios"].items():
            hists = []
            for label, reader in samples:
                hn, hd = reader.get(folder, num), reader.get(folder, den)
                hists.append((ratio_hist(hn, hd, ytitle) if hn and hd else None, label))
            png = "%s__%s.png" % (prefix, outname)
            if draw_overlay(hists, "%s: %s" % (ftitle, outname), os.path.join(opts.out, png)):
                made.append((ftitle, png))
        # Direct distributions.
        for dname in spec["dists"]:
            hists = [(reader.get(folder, dname), label) for label, reader in samples]
            # normalize distributions for shape comparison across samples
            for h, _ in hists:
                if h and h.Integral() > 0:
                    h.Scale(1.0 / h.Integral())
            png = "%s__%s.png" % (prefix, dname)
            if draw_overlay(hists, "%s: %s" % (ftitle, dname), os.path.join(opts.out, png)):
                made.append((ftitle, png))

    # index.html grouped by folder.
    groups = {}
    for ftitle, png in made:
        groups.setdefault(ftitle, []).append(png)
    with open(os.path.join(opts.out, "index.html"), "w", encoding="utf-8") as f:
        f.write("<!doctype html><html><head><meta charset='utf-8'>")
        f.write("<title>Truth-graph validation plots</title>")
        f.write("<style>body{font-family:system-ui,sans-serif;margin:2rem}"
                "h2{margin-top:2rem}img{border:1px solid #ddd;margin:4px}</style></head><body>")
        f.write("<h1>Truth-graph (Branch) validation plots</h1>")
        f.write("<p>Samples: %s</p>" % ", ".join(label for label, _ in samples))
        for ftitle, pngs in groups.items():
            f.write("<h2>%s</h2>" % ftitle)
            for png in pngs:
                f.write("<a href='%s'><img src='%s' width='360'></a>" % (png, png))
        f.write("</body></html>")

    print("Wrote %d plots + index.html to %s/" % (len(made), opts.out))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="+", metavar="FILE[:LABEL]",
                        help="DQM file(s); optional :LABEL for the legend (e.g. ttbar.root:TTbar).")
    parser.add_argument("-o", "--out", default="truthGraphValidationPlots", help="output directory")
    main(parser.parse_args())
