#!/usr/bin/env python3
"""C++ vs Python parity of the exported T4 DNN: the model evaluated on the T4 training-build ntuple (inputs from the
t4x_* branches) vs the softmax outputs (t4_fakeScore, t4_displacedScore) written by a candidate build run on
the same input events (--t4 --t4dnn). T4s are matched by (sim fingerprint, positions of their 8 hits: inner T3 slots
0-5 and outer T3 slots 4-5). Every T4 of the candidate build also exists in the training file, because the candidate
build only adds a DNN cut that removes T4s."""
import argparse
import importlib.util
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("t4train", os.path.join(HERE, "train_T4_DNN.py"))
t4 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t4)


def keys(fn):
    import awkward as ak
    import uproot
    br = [f"t4_t3_{h}_{c}" for h in range(6) for c in "xyz"]
    a = uproot.open(fn)["tree"].arrays(["t4_t3_idx0", "t4_t3_idx1", "t4_fakeScore", "t4_displacedScore",
                                        "sim_pt"] + br)
    ks, sc = [], []
    for e in range(len(a)):
        sp = a["sim_pt"][e].tolist()
        evk = (len(sp),) + tuple(round(x, 4) for x in sp[:3])
        i0, i1 = a["t4_t3_idx0"][e], a["t4_t3_idx1"][e]
        cols = [ak.to_numpy(a[f"t4_t3_{h}_{c}"][e][idx]) for idx, hs in ((i0, range(6)), (i1, range(4, 6)))
                for h in hs for c in "xyz"]
        P = np.round(np.stack(cols, axis=1), 3) if len(i0) else np.zeros((0, 24))
        ks += [(evk,) + tuple(r) for r in P.tolist()]
        sc += np.stack([ak.to_numpy(a[k][e]) for k in ("t4_fakeScore", "t4_displacedScore")],
                       axis=1).tolist()
    return ks, np.array(sc), len(a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--train-file", required=True)
    ap.add_argument("--cand-file", required=True)
    a = ap.parse_args()
    import torch
    ck = torch.load(a.model, map_location="cpu", weights_only=False)
    names, mu, sd = ck["feature_names"], np.array(ck["mean"]), np.array(ck["std"])
    model = torch.nn.Sequential(torch.nn.Linear(len(names), 32), torch.nn.ReLU(), torch.nn.Linear(32, 32),
                                torch.nn.ReLU(), torch.nn.Linear(32, 3))
    model.load_state_dict(ck["state_dict"])
    model.eval()
    ckeys, csc, ncand = keys(a.cand_file)
    cand = dict(zip(ckeys, csc))
    assert len(cand) == len(ckeys), "candidate keys not unique"
    H, f, evt, _ = t4.load_sample(a.train_file)
    X = t4.build_features(H, f)
    Xm = np.stack([X[n] for n in names], axis=1)
    with torch.no_grad():
        p = np.array(torch.softmax(model(torch.tensor(np.ascontiguousarray(((Xm - mu) / sd).astype(np.float32)))),
                                   dim=1).tolist())
    tk, _, _ = keys(a.train_file)
    assert len(tk) == len(p)
    py = dict(zip(tk, p[:, [0, 2]]))  # fake, displaced (prompt = 1 - both; t4_promptScore is -d only)
    common = [k for k in cand if k in py]
    print(f"candidate T4s {len(cand)} in {ncand} events, training-file T4s {len(py)}, matched {len(common)}")
    assert common
    d = np.array([np.abs(cand[k] - py[k]).max() for k in common])
    fin = np.isfinite(d)
    print(f"|C++ - python| (max over fake, displaced): max {d[fin].max():.2e}, frac < 1e-4 {np.mean(d[fin] < 1e-4):.6f}, "
          f"frac < 1e-3 {np.mean(d[fin] < 1e-3):.6f}, non-finite {(~fin).sum()}")


if __name__ == "__main__":
    main()
