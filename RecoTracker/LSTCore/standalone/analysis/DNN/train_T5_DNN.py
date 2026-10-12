#!/usr/bin/env python3
"""Train the 3-class T5 DNN (fake / prompt / displaced) and write the model for wp_T5_DNN.py (replaces train_T5_DNN.ipynb).

Training samples: build the standalone with -d (CUT_VALUE_DEBUG, which writes the t5x_* DNN-input branches) and with
the T5 DNN creation cut in runQuintupletDefaultAlgo (Quintuplet.h, "if (!inference) return false") commented out, so the
sample is the un-selected T5 population; run lst_cpu with --t5 --t5dnn (100-event chunks). Samples used in 2026-09:
PU200 ttbar 1000 events (reference), jets 500 events (-J), 10-muon 0.5-50 GeV 5 cm cube gun 10k events.

Recipe:
  * samples: a reference sample plus enrichment samples, each with a share of the total train loss weight
    (--share NAME=frac); a --flat sample gets weight 1 per row before its share is applied;
  * labels: fake (t5_isFake), otherwise displaced if sim vxy >= --vxy-split (1 cm), else prompt;
  * class weights per sample: fakes 1, trues n_fake/n_true; on the reference sample only, displaced trues are
    weighted x8 (1 <= vxy < 5 cm) and x16 (vxy >= 5 cm) unless --no-tier;
  * 60/20/20 split BY EVENT per sample (seed 42), standardisation fitted on the reference train split only
    (folded into layer 1 by wp_T5_DNN.py --export);
  * n -> 32 -> 32 -> 3, ReLU, cross-entropy with per-row weights, Adam 3e-3 cosine to 1e-5, 300 epochs, batch 16384;
  * model selection: min over samples of AUC(prompt vs fake) and AUC(displaced vs fake), a sample counting only if its
    val split has >= --sel-min-rows rows per class.
Inputs (feature groups, --groups; all by default = the shipped 35): base = the 23 geometric inputs (anchor-hit
eta/phi/z/r and deltas, log10 radii), t3raw = the 6 parent-T3 DNN scores, mddir = MD-direction log-LR mean/max,
density = log10(1+x) of T3s leaving the middle / first MD and MDs in the first module, dca = clipped log10(1+dcaXY).
All are rebuilt exactly as NeuralNetwork.h t5dnn::runInference computes them; before training the script checks that
the current weights header, evaluated on the rebuilt inputs, reproduces the written t5_dnnScore (= 1 - P(fake)).
Permutation importance per feature and group is saved; use drop-one-group retrains (several seeds) to check that each
input earns its place.

Example (2026-09 production model):
  python3 train_T5_DNN.py --lab pu='samples/pu/chunk_*.root' --lab jet='samples/jet/chunk_*.root' \
      --lab gun='samples/gun/chunk_*.root' --share jet=0.25 --share gun=0.02 --flat gun --ref pu --tag sel
"""
import argparse
import glob
import json
import math
import os
import time
import types

import numpy as np

T0 = time.time()

KETA_NORM, KPHI_NORM, KZ_MAX, KR_MAX = 2.5, math.pi, 267.2349854, 110.1099396  # dnn::t5dnn constants (Common.h)
T5X = ["mdDirMeanW", "mdDirMaxW", "nT3OutMid", "nT3OutFirst", "nMDFirstMod", "dcaXY"]  # t5x_* debug branches
BASE = ["eta1", "absPhi1", "z1", "r1"] + [f"{q}{b}{a}" for a, b in [(1, 2), (2, 3), (3, 4), (4, 5)]
                                          for q in ("dEta", "dPhi", "dZ", "dR")] + ["logRin", "logRbridge", "logRout"]
GROUPS = {
    "base": BASE,
    "t3raw": ["fake1", "prompt1", "disp1", "fake2", "prompt2", "disp2"],
    "mddir": ["mdDirMeanW", "mdDirMaxW"],
    "density": ["log1p_nT3OutMid", "log1p_nT3OutFirst", "log1p_nMDFirstMod"],
    "dca": ["log1p_dcaXY"],
}
SHIPPED = [n for g in GROUPS.values() for n in g]  # input order of t5dnn::runInference


def log(m):
    print("[%7.1fs] %s" % (time.time() - T0, m), flush=True)


def load_sample(pattern):
    """Per-T5 feature table, labels and event ids for all chunks matching pattern."""
    import awkward as ak
    import uproot
    files = sorted(glob.glob(pattern))
    assert files, f"no files for {pattern}"
    t5b = ["t5_isFake", "t5_sim_vxy", "t5_pt", "t5_eta", "t5_phi", "t5_innerRadius", "t5_bridgeRadius",
           "t5_outerRadius", "t5_t3_idx0", "t5_t3_idx1", "t5_dnnScore", "t5_isDuplicate", "t5_pMatched"] + \
          [f"t5_t3_{s}Score{i}" for i in (1, 2) for s in ("fake", "prompt", "displaced")] + [f"t5x_{n}" for n in T5X]
    t3b = [f"t5_t3_{h}_{p}" for h in (0, 2, 4) for p in ("eta", "phi", "z", "r")]
    cols, prov = [], []
    for ic, fn in enumerate(files):
        done = fn + ".done"
        prov.append(open(done).read().strip() if os.path.exists(done) else "NO .done FILE")
        a = uproot.open(fn)["tree"].arrays(t5b + t3b)
        n = ak.num(a["t5_pt"])
        i0, i1 = a["t5_t3_idx0"], a["t5_t3_idx1"]
        # five anchor hits: inner T3 slots 0,2,4 and outer T3 slots 2,4
        H = {}
        for k, (idx, slot) in enumerate([(i0, 0), (i0, 2), (i0, 4), (i1, 2), (i1, 4)]):
            for p in ("eta", "phi", "z", "r"):
                H[(k, p)] = ak.to_numpy(ak.flatten(a[f"t5_t3_{slot}_{p}"][idx])).astype(np.float64)
        f = {k: ak.to_numpy(ak.flatten(a[k])) for k in t5b}
        # alignment check: T5-level phi is the phi of anchor hit 1 or 2 (layer-2 adjustment)
        ph = f["t5_phi"]
        ok = np.isclose(ph, H[(0, "phi")], atol=1e-4) | np.isclose(ph, H[(1, "phi")], atol=1e-4)
        assert ok.mean() > 0.999, f"T5/T3 branch alignment broken in {fn}: {ok.mean():.4f}"
        # outer T3 shares the middle MD: its first anchor is the inner T3's last anchor
        ok = np.isclose(H[(2, "r")], ak.to_numpy(ak.flatten(a["t5_t3_0_r"][i1])), atol=1e-4)
        assert ok.mean() > 0.999, f"outer-T3 index alignment broken in {fn}: {ok.mean():.4f}"
        evt = np.repeat(np.arange(len(n)) + ic * 100000, ak.to_numpy(n))
        cols.append((H, f, evt))
        log(f"  {os.path.basename(fn)}: {len(evt)} T5s, {len(n)} events")
    H = {k: np.concatenate([c[0][k] for c in cols]) for k in cols[0][0]}
    f = {k: np.concatenate([c[1][k] for c in cols]) for k in cols[0][1]}
    evt = np.concatenate([c[2] for c in cols])
    return H, f, evt, prov


def build_features(H, f):
    """Named raw features; base ones reproduce NeuralNetwork.h t5dnn::runInference exactly."""
    X = {}
    eta = [np.abs(H[(k, "eta")]) for k in range(5)]
    phi = [H[(k, "phi")] for k in range(5)]
    z = [np.abs(H[(k, "z")]) for k in range(5)]
    r = [H[(k, "r")] for k in range(5)]
    X["eta1"] = eta[0] / KETA_NORM
    X["absPhi1"] = np.abs(phi[0]) / KPHI_NORM
    X["z1"] = z[0] / KZ_MAX
    X["r1"] = r[0] / KR_MAX
    for a in range(4):
        b = a + 1
        dphi = np.remainder(phi[b] - phi[a] + math.pi, 2 * math.pi) - math.pi  # cms::alpakatools::deltaPhi
        X[f"dEta{b + 1}{a + 1}"] = eta[b] - eta[a]
        X[f"dPhi{b + 1}{a + 1}"] = dphi / KPHI_NORM
        X[f"dZ{b + 1}{a + 1}"] = (z[b] - z[a]) / KZ_MAX
        X[f"dR{b + 1}{a + 1}"] = (r[b] - r[a]) / KR_MAX
    X["logRin"] = np.log10(f["t5_innerRadius"])
    X["logRbridge"] = np.log10(f["t5_bridgeRadius"])
    X["logRout"] = np.log10(f["t5_outerRadius"])
    s = {k: f[f"t5_t3_{c}Score{i}"] for k, c, i in [("fake1", "fake", 1), ("prompt1", "prompt", 1),
                                                     ("disp1", "displaced", 1), ("fake2", "fake", 2),
                                                     ("prompt2", "prompt", 2), ("disp2", "displaced", 2)]}
    X.update(s)
    X["mdDirMeanW"] = f["t5x_mdDirMeanW"]
    X["mdDirMaxW"] = f["t5x_mdDirMaxW"]
    for n in ("nT3OutMid", "nT3OutFirst", "nMDFirstMod"):
        X["log1p_" + n] = np.log10(1.0 + f["t5x_" + n])
    X["log1p_dcaXY"] = np.clip(np.log10(1.0 + f["t5x_dcaXY"]), 0.0, 1.4913617)  # log10(1 + 30 cm), dnn::t5dnn::kLogDcaMax
    return X


def read_header_mlp(header):
    """(W1, b1, W2, b2, W3, b3) of an LST weights header (wgtT_* are [in][out])."""
    import re
    txt = open(header).read()

    def arr(name):
        m = re.search(name + r"\[(\d+)\](?:\[(\d+)\])?\s*=\s*\{(.*?)\};", txt, re.S)
        v = np.array([float(x) for x in re.findall(r"-?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?", m.group(3))])
        return v.reshape(int(m.group(1)), int(m.group(2))) if m.group(2) else v
    return [arr(n) for n in ("wgtT_layer1", "bias_layer1", "wgtT_layer2", "bias_layer2", "wgtT_output_layer",
                             "bias_output_layer")]


def softmax_mlp(x, layers):
    w1, b1, w2, b2, w3, b3 = layers
    z = np.maximum(np.maximum(x @ w1 + b1, 0) @ w2 + b2, 0) @ w3 + b3
    p = np.exp(z - z.max(axis=1, keepdims=True))
    return p / p.sum(axis=1, keepdims=True)


def current_dnn_parity(X, f, header):
    """The current T5 DNN (header, SHIPPED inputs) on the rebuilt features must reproduce t5_dnnScore = 1 - P(fake)."""
    p = softmax_mlp(np.stack([X[n] for n in SHIPPED], axis=1), read_header_mlp(header))
    d = np.abs(1.0 - p[:, 0] - f["t5_dnnScore"])
    fin = np.isfinite(d)
    return float(d[fin].max()), float((d[fin] < 1e-4).mean())


def main(spec=None):
    """spec: object description (PREFIX, GROUPS, load_sample, build_features, parity, WEIGHTS); default T5."""
    spec = spec or T5
    P = spec.PREFIX
    ap = argparse.ArgumentParser()
    ap.add_argument("--lab", action="append", required=True, help="NAME=glob of training-build chunks")
    ap.add_argument("--ref", default="pu")
    ap.add_argument("--share", action="append", default=[], help="NAME=frac of total train loss weight")
    ap.add_argument("--flat", action="append", default=[], help="sample with weight 1 per row (before its share)")
    ap.add_argument("--groups", default=",".join(spec.GROUPS), help="feature groups to use")
    ap.add_argument("--vxy-split", type=float, default=1.0, help="displaced iff true and sim vxy >= this [cm]")
    ap.add_argument("--no-tier", action="store_true", help="disable the x8/x16 displaced tiers on the ref sample")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--lr-min", type=float, default=1e-5)
    ap.add_argument("--batch", type=int, default=16384)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sel-min-rows", type=int, default=500)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="models")
    ap.add_argument("--weights", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                     "../../../src/alpaka/" + spec.WEIGHTS),
                    help="current weights header, for the parity check of the rebuilt inputs")
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()

    import torch
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
    groups = [g for g in a.groups.split(",") if g]
    assert all(g in spec.GROUPS for g in groups), groups
    names = [n for g in groups for n in spec.GROUPS[g]]
    labs = dict(x.split("=", 1) for x in a.lab)
    shares = {k: float(v) for k, v in (x.split("=", 1) for x in a.share)}
    assert a.ref in labs and a.ref not in shares and sum(shares.values()) < 1
    log(f"groups {groups} -> {len(names)} inputs; samples {list(labs)}; shares {shares}; flat {a.flat}")

    S = {}
    for nm, pat in labs.items():
        log(f"loading {nm}: {pat}")
        H, f, evt, prov = spec.load_sample(pat)
        X = spec.build_features(H, f)
        dmax, fok = spec.parity(X, f, a.weights)
        log(f"  {nm}: current-DNN parity on the rebuilt inputs: max |diff| {dmax:.2e}, {fok:.5f} within 1e-4")
        assert fok > 0.999, "rebuilt inputs do not reproduce the current DNN"
        Xm = np.stack([X[n] for n in names], axis=1).astype(np.float32)
        true = f[P + "isFake"] == 0
        vxy = f[P + "sim_vxy"]
        y = np.where(~true, 0, np.where(vxy >= a.vxy_split, 2, 1)).astype(np.int64)
        good = np.isfinite(Xm).all(axis=1)
        if (~good).any():
            log(f"  {nm}: dropping {(~good).sum()} rows with non-finite inputs")
        ev = np.unique(evt)
        perm = rng.permutation(ev)
        ntr, nva = int(0.6 * len(ev)), int(0.2 * len(ev))
        part = np.zeros(len(evt), np.int8)  # 0 train, 1 val, 2 test
        part[np.isin(evt, perm[ntr:ntr + nva])] = 1
        part[np.isin(evt, perm[ntr + nva:])] = 2
        S[nm] = dict(X=Xm[good], y=y[good], vxy=vxy[good], part=part[good], evt=evt[good], prov=prov,
                     pt=f[P + "pt"][good])
        c = np.bincount(S[nm]["y"], minlength=3)
        log(f"  {nm}: rows {len(S[nm]['y'])} (fake {c[0]}, prompt {c[1]}, displaced {c[2]}), events {len(ev)}")

    # per-row weights: class balance (fakes 1, trues n_fake/n_true), ref-only displaced tiers, then shares
    W = {}
    for nm, s in S.items():
        tr = s["part"] == 0
        w = np.ones(len(s["y"]), np.float64)
        if nm not in a.flat:
            nf, nt = (s["y"][tr] == 0).sum(), (s["y"][tr] > 0).sum()
            w[s["y"] > 0] = nf / max(nt, 1)
            if nm == a.ref and not a.no_tier:
                d = s["y"] == 2
                w[d & (s["vxy"] < 5.0)] *= 8.0
                w[d & (s["vxy"] >= 5.0)] *= 16.0
        W[nm] = w
    wref = W[a.ref][S[a.ref]["part"] == 0].sum()
    sum_sh = sum(shares.values())
    for nm in S:
        if nm == a.ref:
            continue
        sh = shares.get(nm, 0.0)
        wk = W[nm][S[nm]["part"] == 0].sum()
        W[nm] *= (sh / (1.0 - sum_sh)) * wref / wk if wk > 0 else 0.0
    tot = {nm: W[nm][S[nm]["part"] == 0].sum() for nm in S}
    T = sum(tot.values())
    log("train loss-weight shares: " + ", ".join(f"{k} {v / T:.3f}" for k, v in tot.items()))
    cls = np.zeros(3)
    for nm in S:
        tr = S[nm]["part"] == 0
        cls += np.bincount(S[nm]["y"][tr], weights=W[nm][tr], minlength=3)
    log("train loss-weight by class (fake/prompt/disp): " + "/".join(f"{c / cls.sum():.3f}" for c in cls))

    # standardisation on the reference train split only
    Xr = S[a.ref]["X"][S[a.ref]["part"] == 0].astype(np.float64)
    mu, sd = Xr.mean(axis=0), Xr.std(axis=0)
    sd[sd < 1e-12] = 1.0

    def T_(x):
        return torch.tensor(np.ascontiguousarray(x))

    Xtr = T_(np.concatenate([(S[nm]["X"][S[nm]["part"] == 0] - mu) / sd for nm in S]).astype(np.float32)).to(dev)
    ytr = T_(np.concatenate([S[nm]["y"][S[nm]["part"] == 0] for nm in S])).to(dev)
    wtr = T_(np.concatenate([W[nm][S[nm]["part"] == 0] for nm in S]).astype(np.float32)).to(dev)
    wtr = wtr / wtr.mean()
    VA = {nm: (T_(((S[nm]["X"][S[nm]["part"] == 1] - mu) / sd).astype(np.float32)).to(dev),
               T_(S[nm]["y"][S[nm]["part"] == 1]).to(dev)) for nm in S}
    log(f"train rows {len(ytr)}; val rows " + ", ".join(f"{k} {len(v[1])}" for k, v in VA.items()))

    model = torch.nn.Sequential(torch.nn.Linear(len(names), a.hidden), torch.nn.ReLU(),
                                torch.nn.Linear(a.hidden, a.hidden), torch.nn.ReLU(),
                                torch.nn.Linear(a.hidden, 3)).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs, eta_min=a.lr_min)
    ce = torch.nn.CrossEntropyLoss(reduction="none")

    def auc(score, pos):
        """ROC AUC with ties averaged; score/pos torch tensors on dev."""
        n1, n0 = int(pos.sum()), int((~pos).sum())
        if n1 == 0 or n0 == 0:
            return float("nan")
        order = torch.argsort(score)
        s = score[order]
        ranks = torch.empty_like(s, dtype=torch.float64)
        ranks[order] = torch.arange(1, len(s) + 1, device=s.device, dtype=torch.float64)
        # average ranks over ties
        uniq, inv, counts = torch.unique(s, return_inverse=True, return_counts=True)
        if len(uniq) < len(s):
            rsum = torch.zeros(len(uniq), dtype=torch.float64, device=s.device).index_add_(0, inv, ranks[order])
            ranks[order] = (rsum / counts)[inv]
        return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

    @torch.no_grad()
    def evaluate(Xv, yv, mdl=None):
        z = (mdl or model)(Xv)
        zp, zd = z[:, 1] - z[:, 0], z[:, 2] - z[:, 0]
        mP, mD = (yv == 1) | (yv == 0), (yv == 2) | (yv == 0)
        return auc(zp[mP], yv[mP] == 1), auc(zd[mD], yv[mD] == 2)

    def selector(res):
        vals = []
        for nm, (aP, aD) in res.items():
            c = torch.bincount(VA[nm][1], minlength=3)
            if int(c.min()) >= a.sel_min_rows:
                vals += [aP, aD]
        return min(vals) if vals else float("nan")

    best, best_state, best_ep, hist = -1.0, None, -1, []
    n = len(ytr)
    for ep in range(a.epochs):
        model.train()
        perm = torch.randperm(n, device=dev)
        tl = torch.zeros((), device=dev)
        for i in range(0, n, a.batch):
            idx = perm[i:i + a.batch]
            loss = (ce(model(Xtr[idx]), ytr[idx]) * wtr[idx]).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tl += loss.detach() * len(idx)
        sched.step()
        tl = float(tl)
        model.eval()
        res = {nm: evaluate(*VA[nm]) for nm in VA}
        sel = selector(res)
        hist.append(dict(ep=ep, loss=tl / n, sel=sel, **{f"{k}_{c}": v for k, (p, d) in res.items()
                                                          for c, v in (("P", p), ("D", d))}))
        if sel > best:
            best, best_ep = sel, ep
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if ep % 10 == 0 or ep == a.epochs - 1:
            log(f"ep {ep:3d} loss {tl / n:.5f} sel {sel:.5f} best {best:.5f}@{best_ep} | " +
                " ".join(f"{k} P{p:.4f} D{d:.4f}" for k, (p, d) in res.items()))
    model.load_state_dict(best_state)
    model.eval()

    # permutation importance on the val splits: drop in each sample's AUCs when a feature / group is shuffled
    gen = torch.Generator(device=dev).manual_seed(a.seed)
    base_res = {nm: evaluate(*VA[nm]) for nm in VA}
    imp = {"feature": {}, "group": {}}

    def shuffled(Xv, cols):
        Xs = Xv.clone()
        p = torch.randperm(len(Xv), device=dev, generator=gen)
        for c in cols:
            Xs[:, c] = Xv[p, c]
        return Xs

    for key, items in (("feature", [[n] for n in names]), ("group", [spec.GROUPS[g] for g in groups])):
        for it in items:
            cols = [names.index(x) for x in it]
            nm_it = it[0] if key == "feature" else [g for g in groups if spec.GROUPS[g] == it][0]
            imp[key][nm_it] = {}
            for nm in VA:
                r = evaluate(shuffled(VA[nm][0], cols), VA[nm][1])
                imp[key][nm_it][nm] = [base_res[nm][0] - r[0], base_res[nm][1] - r[1]]
    log("group permutation importance (AUC drop P/D per sample):")
    for g, d in imp["group"].items():
        log(f"  {g:8s} " + " ".join(f"{k} {v[0]:+.4f}/{v[1]:+.4f}" for k, v in d.items()))

    os.makedirs(a.out, exist_ok=True)
    meta = dict(tag=a.tag, args=vars(a), feature_names=names, groups=groups, mean=mu.tolist(), std=sd.tolist(),
                arch=[len(names), a.hidden, a.hidden, 3], classes=["fake", "prompt", "displaced"],
                best_epoch=best_ep, best_sel=best, val=base_res, importance=imp,
                provenance={nm: S[nm]["prov"] for nm in S}, history=hist,
                constants=dict(eta_norm=KETA_NORM, phi_norm=KPHI_NORM, z_max=KZ_MAX, r_max=KR_MAX))
    torch.save(dict(state_dict=model.state_dict(), **{k: v for k, v in meta.items() if k != "history"}),
               os.path.join(a.out, f"{P[:-1]}dnn_{a.tag}.pt"))
    json.dump(meta, open(os.path.join(a.out, f"{P[:-1]}dnn_{a.tag}.json"), "w"), indent=1)
    log(f"saved {a.out}/{P[:-1]}dnn_{a.tag}.pt ; best epoch {best_ep}, selector {best:.5f}")


T5 = types.SimpleNamespace(PREFIX="t5_", GROUPS=GROUPS, load_sample=load_sample, build_features=build_features,
                           parity=current_dnn_parity, WEIGHTS="T5NeuralNetworkWeights.h")

if __name__ == "__main__":
    main()
