#!/usr/bin/env python3
"""C++ vs Python parity of an exported T5 DNN.

Evaluates the trained model (train_T5_DNN.py .pt) on a training-sample chunk (inputs from the t5x_* branches) and
compares 1 - P(fake) to t5_dnnScore written by a build with the exported header, run on the same input events (e.g.
the same --nsplit_jobs/--job_index and a smaller -n). T5s are matched by (sim fingerprint, positions of the 10 hits),
so stream order does not matter. Every T5 of the build also exists in the training sample, because the build only
adds a DNN cut that removes T5s.
  python3 parity_T5_DNN.py --model models/t5dnn_sel.pt --train-file samples/pu/chunk_0.root --cand-file cand.root
"""
import argparse
import importlib.util
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("tr", os.path.join(HERE, "train_T5_DNN.py"))
tr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tr)


def t5keys(fn):
    """(event fingerprint, positions of the 10 hits) per T5, in file order. A T5 is defined by its 5 MDs (an anchor
    hit alone can belong to several MDs); the writer's T3 row indices are not stable across builds. The fingerprint (sim count + leading
    sim pts) removes the stream-dependent entry order."""
    import awkward as ak
    import uproot
    br = [f"t5_t3_{h}_{c}" for h in range(6) for c in "xyz"]
    a = uproot.open(fn)["tree"].arrays(["t5_t3_idx0", "t5_t3_idx1", "t5_dnnScore", "sim_pt"] + br)
    ks, sc = [], []
    for e in range(len(a)):
        sp = a["sim_pt"][e].tolist()
        evk = (len(sp),) + tuple(round(x, 4) for x in sp[:3])
        i0, i1 = a["t5_t3_idx0"][e], a["t5_t3_idx1"][e]
        cols = [ak.to_numpy(a[f"t5_t3_{h}_{c}"][e][idx]) for idx, hs in ((i0, range(6)), (i1, range(2, 6)))
                for h in hs for c in "xyz"]
        P = np.round(np.stack(cols, axis=1), 3)
        ks += [(evk,) + tuple(r) for r in P.tolist()]
        sc += a["t5_dnnScore"][e].tolist()
    return ks, np.array(sc), len(a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--train-file", required=True, help="training-build chunk holding the same events")
    ap.add_argument("--cand-file", required=True)
    a = ap.parse_args()
    import torch
    ck = torch.load(a.model, map_location="cpu", weights_only=False)
    names, mu, sd = ck["feature_names"], np.array(ck["mean"]), np.array(ck["std"])
    model = torch.nn.Sequential(torch.nn.Linear(len(names), ck["arch"][1]), torch.nn.ReLU(),
                                torch.nn.Linear(ck["arch"][1], ck["arch"][2]), torch.nn.ReLU(),
                                torch.nn.Linear(ck["arch"][2], 3))
    model.load_state_dict(ck["state_dict"])
    model.eval()
    ck_, cs, ncand = t5keys(a.cand_file)
    cand = dict(zip(ck_, cs))
    assert len(cand) == len(ck_), "candidate keys not unique"
    # candidate events are a subset of the chunk's events in unknown entry order: match over the whole chunk
    H, f, evt, _ = tr.load_sample(a.train_file)
    X = tr.build_features(H, f)
    Xm = np.stack([X[n] for n in names], axis=1)
    with torch.no_grad():
        p = np.array(torch.softmax(model(torch.tensor(np.ascontiguousarray(((Xm - mu) / sd).astype(np.float32)))),
                                   dim=1).tolist())
    tk, _, _ = t5keys(a.train_file)
    assert len(tk) == len(p)
    py = dict(zip(tk, 1.0 - p[:, 0]))
    assert len(py) == len(tk), "training keys not unique"
    common = [k for k in cand if k in py]
    d = np.array([abs(cand[k] - py[k]) for k in common])
    fin = np.isfinite(d)
    print(f"candidate T5s {len(cand)} in {ncand} events, training-file T5s {len(py)}, matched {len(common)}")
    assert common, "no matched T5s"
    print(f"|C++ - python| on matched: max {d[fin].max():.2e}, frac < 1e-4 {np.mean(d[fin] < 1e-4):.6f}, "
          f"frac < 1e-3 {np.mean(d[fin] < 1e-3):.6f}, non-finite {(~fin).sum()}")


if __name__ == "__main__":
    main()
