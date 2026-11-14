#!/usr/bin/env python3
"""Working point and weights header for a T5 DNN trained by train_T5_DNN.py.

The T5 creation cut (t5dnn::runInference) is one per-bin table on 1 - P(fake): pass = 1 - P(fake) > kWp[pt][eta],
pt bin = (innerRadius * k2Rinv1GeVf * 2 > 5 GeV), eta bin = 0.25-wide on |eta| of the first anchor hit (last bin open).
A table with retention r keeps the fraction r of FULLY matched (pMatched > 0.95) T5s per bin, derived on the held-out
(val + test) events of the reference sample of the training split. At equal fake retention this beat separate
prompt/displaced tables (an OR spends the fake budget on displaced). The retention is chosen on the deployed
efficiency / fake / duplicate rates (2026-09: r = 0.95, scanned 0.90-0.99); the kept fractions printed here orient only.

Printed per sample: kept fraction of fakes / prompt / displaced bands for the current table (t5_dnnScore vs kWp in
Common.h) and for each new retention. --export writes <out>_T5NeuralNetworkWeights.h (standardisation folded into
layer 1) and <out>_kWp.h; install both with install_T5_WP.py.
  python3 wp_T5_DNN.py --model models/t5dnn_sel.pt --out wp/sel.json --export 0.95
"""
import argparse
import importlib.util
import json
import os
import re

import numpy as np

K2RINV1GEV2 = 2.99792458e-3 * 3.8  # k2Rinv1GeVf * 2
ETA_SIZE, NETA = 0.25, 10
VXY_BANDS = [(1.0, 5.0), (5.0, 25.0), (25.0, 1e9)]
HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("tr", os.path.join(HERE, "train_T5_DNN.py"))
tr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tr)


def bins(pt, eta):
    return (pt > 5.0).astype(int), np.minimum((np.abs(eta) / ETA_SIZE).astype(int), NETA - 1)


def table(p, sel, ptb, etab, ret):
    """Per-bin threshold keeping fraction ret of the selected rows (counts returned for the record)."""
    wp, n = np.zeros((2, NETA)), np.zeros((2, NETA), int)
    for i in range(2):
        for j in range(NETA):
            m = sel & (ptb == i) & (etab == j)
            n[i, j] = m.sum()
            wp[i, j] = np.quantile(p[m], 1.0 - ret) if n[i, j] else 0.0
    return wp, n


def current_table(common_h):
    blk = open(common_h).read()
    blk = blk[blk.index("namespace t5dnn"):]
    m = re.search(r"kWp\[kPtBins\]\[kEtaBins\]\s*=\s*\{(.*?)\};", blk, re.S)
    return np.array([float(v) for v in re.findall(r"-?[0-9]*\.?[0-9]+", m.group(1))]).reshape(2, NETA)


def write_md(res, path, title, cut_text, keys):
    """Markdown record of a WP derivation: every table (paste-ready kWp, per-bin row counts) and the held-out kept
    fractions of every rule (current cut and each table) per sample, as the notebooks' printouts used to keep."""
    L = [f"# {title}", "", f"Model: `{res['model']}`", "", cut_text, "",
         "Bins: pt bin 0 = pt < 5 GeV, 1 = pt > 5 GeV; eta bins of 0.25 in |eta| of the first anchor hit (last open).",
         "", "## Held-out kept fractions (kept/total)", ""]
    for smp, R in res["samples"].items():
        cls = list(next(iter(R.values())).keys())
        L += [f"### {smp}", "", "| rule | " + " | ".join(cls) + " |", "|---" * (len(cls) + 1) + "|"]
        for rule, d in R.items():
            L.append(f"| {rule} | " + " | ".join(f"{d[c][0] / max(d[c][1], 1):.4f}" for c in cls) + " |")
        L.append("")
    L += ["## Tables", ""]
    for k in keys:
        t = res["tables"][k]
        rows = ",\n    ".join("{" + ", ".join(f"{x:.4f}f" for x in r) + "}" for r in t["wp"])
        L += [f"### {k}", "", "```cpp", f"HOST_DEVICE_CONSTANT float kWp[kPtBins][kEtaBins] = {{\n    {rows}}};", "```",
              "", "Fully matched rows per bin (the quantile statistics): " +
              " / ".join(",".join(str(int(x)) for x in r) for r in t["n"]), ""]
    open(path, "w").write("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--retention", type=float, nargs="+", default=[0.9, 0.93, 0.95, 0.97, 0.98, 0.99])
    ap.add_argument("--common-h", default=os.path.join(HERE, "../../../interface/alpaka/Common.h"))
    ap.add_argument("--export", type=float, help="write the weights header + the kWp table for this retention")
    ap.add_argument("--out", required=True)
    ap.add_argument("--md", help="also write a markdown record of all tables and kept fractions")
    a = ap.parse_args()
    import torch
    ck = torch.load(a.model, map_location="cpu", weights_only=False)
    targs, names = ck["args"], ck["feature_names"]
    mu, sd = np.array(ck["mean"]), np.array(ck["std"])
    labs = dict(x.split("=", 1) for x in targs["lab"])
    rng = np.random.default_rng(targs["seed"])
    model = torch.nn.Sequential(torch.nn.Linear(len(names), ck["arch"][1]), torch.nn.ReLU(),
                                torch.nn.Linear(ck["arch"][1], ck["arch"][2]), torch.nn.ReLU(),
                                torch.nn.Linear(ck["arch"][2], 3))
    model.load_state_dict(ck["state_dict"])
    model.eval()
    cur = current_table(a.common_h)
    split = targs["vxy_split"]
    res, tabs = dict(model=a.model, retention=a.retention, samples={}), {}
    for nm, pat in labs.items():  # same order and rng draws as the training split
        H, f, evt, _ = tr.load_sample(pat)
        ev = np.unique(evt)
        perm = rng.permutation(ev)
        if nm != targs["ref"] and nm != "jet":
            continue
        held = np.isin(evt, perm[int(0.6 * len(ev)):])
        X = tr.build_features(H, f)
        Xm = np.stack([X[n] for n in names], axis=1)
        g = held & np.isfinite(Xm).all(axis=1)
        with torch.no_grad():
            p = np.array(torch.softmax(model(torch.tensor(np.ascontiguousarray(((Xm[g] - mu) / sd).astype(np.float32)))),
                                       dim=1).tolist())
        score = 1.0 - p[:, 0]
        ptb, eb = bins(f["t5_innerRadius"][g] * K2RINV1GEV2, H[(0, "eta")][g])
        vxy, fake, full = f["t5_sim_vxy"][g], f["t5_isFake"][g] == 1, f["t5_pMatched"][g] > 0.95
        if nm == targs["ref"]:
            for r in a.retention:
                tabs[r] = dict(zip(("wp", "n"), table(score, full, ptb, eb, r)))
        classes = {"fake": fake, "prompt": ~fake & (vxy < split)}
        classes.update({f"disp{lo:g}-{hi:g}": ~fake & (vxy >= lo) & (vxy < hi) for lo, hi in VXY_BANDS})
        passes = {"current": f["t5_dnnScore"][g] > cur[ptb, eb]}
        passes.update({f"new_{r}": score > tabs[r]["wp"][ptb, eb] for r in a.retention})
        R = {k: {c: [int((ps & m).sum()), int(m.sum())] for c, m in classes.items()} for k, ps in passes.items()}
        res["samples"][nm] = R
        print(f"== {nm} held-out T5s {g.sum()}: kept fraction")
        print(f"{'':12s}" + "".join(f"{c:>14s}" for c in classes))
        for k, d in R.items():
            print(f"{k:12s}" + "".join(f"{d[c][0] / max(d[c][1], 1):14.4f}" for c in classes))
    res["tables"] = {str(k): {kk: np.asarray(vv).tolist() for kk, vv in v.items()} for k, v in tabs.items()}
    json.dump(res, open(a.out, "w"), indent=1)
    print("wrote", a.out)
    if a.md:
        write_md(res, a.md, "T5 DNN working points", "Cut: pass = 1 - P(fake) > kWp[pt][eta]; a table with retention r keeps the fraction r of fully matched (pMatched > 0.95, i.e. all 10 hits from one simulated particle) T5s per bin. The fake column uses the standard 75% matching (t5_isFake).", [str(float(r)) for r in a.retention])
        print("wrote", a.md)
    if a.export:
        sdict = ck["state_dict"]
        W1, b1 = np.array(sdict["0.weight"].double().tolist()), np.array(sdict["0.bias"].double().tolist())
        # fold the standardisation: W1 (x - mu) / sd + b1 = (W1 / sd) x + (b1 - W1 mu / sd)
        L = [(W1 / sd, b1 - (W1 * (mu / sd)).sum(axis=1))] + \
            [(np.array(sdict[f"{i}.weight"].double().tolist()), np.array(sdict[f"{i}.bias"].double().tolist()))
             for i in (2, 4)]

        def carr(v):
            return ", ".join(f"{x:.8g}f" for x in v)
        nin, nh = len(names), ck["arch"][1]
        out = ["#ifndef RecoTracker_LSTCore_src_alpaka_T5NeuralNetworkWeights_h",
               "#define RecoTracker_LSTCore_src_alpaka_T5NeuralNetworkWeights_h", "", "#include <alpaka/alpaka.hpp>", "",
               "#include \"FWCore/Utilities/interface/HostDeviceConstant.h\"", "",
               "namespace ALPAKA_ACCELERATOR_NAMESPACE::lst::dnn::t5dnn {"]
        for nmL, (W, b), (i, o) in zip(("layer1", "layer2", "output_layer"), L, ((nin, nh), (nh, nh), (nh, 3))):
            out.append(f"  HOST_DEVICE_CONSTANT float bias_{nmL}[{o}] = {{{carr(b)}}};")
            out.append(f"  HOST_DEVICE_CONSTANT float wgtT_{nmL}[{i}][{o}] = {{")
            out += [f"      {{{carr(row)}}}," for row in W.T]
            out.append("  };")
        out += ["}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::lst::dnn::t5dnn", "", "#endif", ""]
        base = os.path.splitext(a.out)[0]
        open(base + "_T5NeuralNetworkWeights.h", "w").write("\n".join(out))
        t = tabs[a.export]["wp"]
        rows = ",\n          ".join("{" + ", ".join(f"{x:.4f}f" for x in t[i]) + "}" for i in range(2))
        open(base + "_kWp.h", "w").write(
            f"      // keeps {a.export} of fully matched T5s per bin\n"
            f"      HOST_DEVICE_CONSTANT float kWp[kPtBins][kEtaBins] = {{\n          {rows}}};\n")
        print("wrote", base + "_T5NeuralNetworkWeights.h and", base + "_kWp.h; inputs in order:", names)


if __name__ == "__main__":
    main()
