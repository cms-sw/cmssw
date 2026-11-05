#!/usr/bin/env python3
"""Train the 3-class T4 DNN with the recipe of train_T5_DNN.py (whose training core, options and outputs it reuses);
replaces train_T4_DNN.ipynb.

Training samples: a -d build (writes the t4x_* DNN-input branches) with the T4 DNN creation cut in
runQuadrupletDefaultAlgo (Quadruplet.h, "if (!inference) return false") commented out, run with --t4 --t4dnn; the same
three samples and shares as for T5. The 2026-09 production model was trained with --no-tier: the extra loss weights
that the T5 recipe gives displaced true objects on the reference sample (x8 for 1 <= vxy < 5 cm, x16 for vxy >= 5 cm)
are off, because the T4 population is already mostly displaced.
Inputs (--groups, all by default = the shipped 33):
    base    = the 21 geometric inputs: anchor-hit eta/phi/z/r and deltas of the 4 anchors (inner T3 slots 0,2,4 +
              outer T3 slot 4); 1/Rin, 1/Rout, Rin/Rout, 1/Rregression, 1/RnonAnchorRegression
    t3raw   = the 6 parent-T3 DNN scores
    mddir   = MD-direction log-likelihood over the 4 MDs (circle through MDs 0,1,3), mean and largest
    density = T3s leaving the shared MD 1 and MD 0; MDs in the first module
    dca     = dcaXY of that circle
The script checks that the current weights header, evaluated on the rebuilt inputs, reproduces the written T4 scores.
  python3 train_T4_DNN.py --lab pu='samples_t4/pu/chunk_*.root' --lab jet='samples_t4/jet/chunk_*.root' \
      --lab gun='samples_t4/gun/chunk_*.root' --share jet=0.25 --share gun=0.02 --flat gun --ref pu --no-tier --tag sel
"""
import glob
import importlib.util
import math
import os
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("t5train", os.path.join(HERE, "train_T5_DNN.py"))
tr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tr)

KETA_NORM, KPHI_NORM, KZ_MAX, KR_MAX = 2.5, math.pi, 267.2349854, 110.1099396  # dnn::t4dnn constants (Common.h)
T4X = ["mdDirMeanW", "mdDirMaxW", "nT3OutMid", "nT3OutFirst", "nMDFirstMod", "dcaXY"]
BASE = ["eta1", "absPhi1", "z1", "r1"] + [f"{q}{b}{a}" for a, b in [(1, 2), (2, 3), (3, 4)]
                                          for q in ("dEta", "dPhi", "dZ", "dR")] + \
       ["invRin", "invRout", "RinOverRout", "invRreg", "invRnonAnchorReg"]
GROUPS = {
    "base": BASE,
    "t3raw": ["fake1", "prompt1", "disp1", "fake2", "prompt2", "disp2"],
    "mddir": ["mdDirMeanW", "mdDirMaxW"],
    "density": ["log1p_nT3OutMid", "log1p_nT3OutFirst", "log1p_nMDFirstMod"],
    "dca": ["log1p_dcaXY"],
}
SHIPPED = [n for g in GROUPS.values() for n in g]  # input order of t4dnn::runInference
# anchor MDs: inner T3 slots 0,2,4 and outer T3 slot 4 (C++: seg1.md0, seg2.md0, seg2.md1, outer seg2.md1)
ANCHORS = [("i0", 0), ("i0", 2), ("i0", 4), ("i1", 4)]


def load_sample(pattern):
    import awkward as ak
    import uproot
    files = sorted(glob.glob(pattern))
    assert files, f"no files for {pattern}"
    t4b = ["t4_isFake", "t4_sim_vxy", "t4_pt", "t4_pMatched", "t4_innerRadius", "t4_outerRadius",
           "t4_regressionRadius", "t4_nonAnchorRegressionRadius", "t4_t3_idx0", "t4_t3_idx1",
           "t4_promptScore", "t4_displacedScore", "t4_fakeScore"] + \
          [f"t4_t3_{s}Score{i}" for i in (1, 2) for s in ("fake", "prompt", "displaced")] + [f"t4x_{n}" for n in T4X]
    t3b = [f"t4_t3_{h}_{p}" for h in (0, 2, 4) for p in ("eta", "phi", "z", "r")]
    cols, prov = [], []
    for ic, fn in enumerate(files):
        done = fn + ".done"
        prov.append(open(done).read().strip() if os.path.exists(done) else "NO .done FILE")
        a = uproot.open(fn)["tree"].arrays(t4b + t3b)
        n = ak.num(a["t4_pt"])
        idx = {"i0": a["t4_t3_idx0"], "i1": a["t4_t3_idx1"]}
        H = {}
        for k, (which, slot) in enumerate(ANCHORS):
            for p in ("eta", "phi", "z", "r"):
                H[(k, p)] = ak.to_numpy(ak.flatten(a[f"t4_t3_{slot}_{p}"][idx[which]])).astype(np.float64)
        # outer T3 starts at the inner T3's second MD: outer slot 0 == inner slot 2
        ok = np.isclose(ak.to_numpy(ak.flatten(a["t4_t3_0_r"][idx["i1"]])), H[(1, "r")], atol=1e-4)
        assert ok.mean() > 0.999, f"T4/T3 index alignment broken in {fn}: {ok.mean():.4f}"
        f = {k: ak.to_numpy(ak.flatten(a[k])) for k in t4b}
        evt = np.repeat(np.arange(len(n)) + ic * 100000, ak.to_numpy(n))
        cols.append((H, f, evt))
        tr.log(f"  {os.path.basename(fn)}: {len(evt)} T4s, {len(n)} events")
    H = {k: np.concatenate([c[0][k] for c in cols]) for k in cols[0][0]}
    f = {k: np.concatenate([c[1][k] for c in cols]) for k in cols[0][1]}
    return H, f, np.concatenate([c[2] for c in cols]), prov


def build_features(H, f):
    X = {}
    eta = [np.abs(H[(k, "eta")]) for k in range(4)]
    phi = [H[(k, "phi")] for k in range(4)]
    z = [np.abs(H[(k, "z")]) for k in range(4)]
    r = [H[(k, "r")] for k in range(4)]
    X["eta1"], X["absPhi1"], X["z1"], X["r1"] = eta[0] / KETA_NORM, np.abs(phi[0]) / KPHI_NORM, z[0] / KZ_MAX, r[0] / KR_MAX
    for a in range(3):
        b = a + 1
        dphi = np.remainder(phi[b] - phi[a] + math.pi, 2 * math.pi) - math.pi
        X[f"dEta{b + 1}{a + 1}"] = eta[b] - eta[a]
        X[f"dPhi{b + 1}{a + 1}"] = dphi / KPHI_NORM
        X[f"dZ{b + 1}{a + 1}"] = (z[b] - z[a]) / KZ_MAX
        X[f"dR{b + 1}{a + 1}"] = (r[b] - r[a]) / KR_MAX
    rin, rout = f["t4_innerRadius"], f["t4_outerRadius"]
    X["invRin"], X["invRout"], X["RinOverRout"] = 1.0 / rin, 1.0 / rout, rin / rout
    X["invRreg"], X["invRnonAnchorReg"] = 1.0 / f["t4_regressionRadius"], 1.0 / f["t4_nonAnchorRegressionRadius"]
    s = {"fake1": f["t4_t3_fakeScore1"], "prompt1": f["t4_t3_promptScore1"], "disp1": f["t4_t3_displacedScore1"],
         "fake2": f["t4_t3_fakeScore2"], "prompt2": f["t4_t3_promptScore2"], "disp2": f["t4_t3_displacedScore2"]}
    X.update(s)
    X["mdDirMeanW"], X["mdDirMaxW"] = f["t4x_mdDirMeanW"], f["t4x_mdDirMaxW"]
    for n in ("nT3OutMid", "nT3OutFirst", "nMDFirstMod"):
        X["log1p_" + n] = np.log10(1.0 + f["t4x_" + n])
    X["log1p_dcaXY"] = np.clip(np.log10(1.0 + f["t4x_dcaXY"]), 0.0, 1.4913617)  # log10(1 + 30 cm), dnn::t5dnn::kLogDcaMax
    return X


def parity(X, f, header):
    """The current T4 DNN (header, SHIPPED inputs) on the rebuilt features must reproduce the written softmax outputs."""
    p = tr.softmax_mlp(np.stack([X[n] for n in SHIPPED], axis=1), tr.read_header_mlp(header))
    old = np.stack([f["t4_fakeScore"], f["t4_promptScore"], f["t4_displacedScore"]], axis=1)
    d = np.abs(p - old).max(axis=1)
    fin = np.isfinite(d)
    return float(d[fin].max()), float((d[fin] < 1e-4).mean())


T4 = types.SimpleNamespace(PREFIX="t4_", GROUPS=GROUPS, load_sample=load_sample, build_features=build_features,
                           parity=parity, WEIGHTS="T4NeuralNetworkWeights.h")

if __name__ == "__main__":
    tr.main(T4)
