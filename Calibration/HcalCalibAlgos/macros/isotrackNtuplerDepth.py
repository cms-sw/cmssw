#!/usr/bin/env python3
import argparse

import awkward as ak
import numpy as np
import pandas as pd
import uproot

PREFIX_MAP = {
    "t_DetIds": "t_eHcal",
    "t_DetIds1": "t_eHcal10",
    "t_DetIds3": "t_eHcal30",
}
BRANCH_PAIRS = {
    "t_DetIds": "t_HitEnergies",
    "t_DetIds1": "t_HitEnergies1",
    "t_DetIds3": "t_HitEnergies3",
}

PU_CONTROL_BRANCHES = [
    "t_Run", "t_Event", "t_nVtx", "t_ieta", "t_iphi", "t_p", "t_pt",
    "t_gentrackP", "t_eMipDR", "t_eHcal", "t_hmaxNearP", "t_emaxNearP",
    "t_hAnnular", "t_eAnnular", "t_rhoh", "t_selectTk", "t_qltyFlag",
    "t_EventWeight",
]

NOPU_BRANCHES = ["t_Event", "t_ieta", "t_iphi", "t_eHcal"]

MERGE_KEYS = ["t_Event", "t_ieta", "t_iphi"]


def unpack_depth(detid):
    detid = ak.values_astype(detid, np.int64)
    new_format = (detid & 0x1000000) != 0
    depth_new = (detid >> 20) & 0xF
    depth_old = (detid >> 14) & 0x1F
    return ak.where(new_format, depth_new, depth_old)


def sum_by_depth(detids, hitenergies, max_depth, prefix):
    depths = unpack_depth(detids)
    out = {}
    for d in range(1, max_depth + 1):
        mask = depths == d
        summed = ak.sum(hitenergies[mask], axis=1)
        out[f"{prefix}_depth{d}"] = ak.to_numpy(summed)
    out[f"{prefix}_depthAll"] = ak.to_numpy(ak.sum(hitenergies, axis=1))
    return out


def read_with_depth(tree, base_branches, max_depth, entry_start=None, entry_stop=None):
    available = set(tree.keys())
    pairs = {k: v for k, v in BRANCH_PAIRS.items() if k in available and v in available}
    base_present = [b for b in base_branches if b in available]

    to_read = base_present + list(pairs.keys()) + list(pairs.values())
    kwargs = {}
    if entry_start is not None:
        kwargs["entry_start"] = entry_start
    if entry_stop is not None:
        kwargs["entry_stop"] = entry_stop
    arrays = tree.arrays(to_read, library="ak", **kwargs)

    result = {b: ak.to_numpy(arrays[b]) for b in base_present}
    for detid_branch, hiten_branch in pairs.items():
        prefix = PREFIX_MAP[detid_branch]
        result.update(sum_by_depth(arrays[detid_branch], arrays[hiten_branch], max_depth, prefix))

    return pd.DataFrame(result)


def pu_colname(name, dfspu_cols, dfsnpu_cols):
    if name in dfspu_cols and name in dfsnpu_cols:
        return f"{name}_x"
    if name in dfspu_cols:
        return name
    return None


def npu_colname(name, dfspu_cols, dfsnpu_cols):
    if name in dfspu_cols and name in dfsnpu_cols:
        return f"{name}_y"
    if name in dfsnpu_cols:
        return name
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-PU", "--filePU", required=True, help="Input PU ROOT file")
    ap.add_argument("-NPU", "--fileNPU", required=True, help="Input no-PU ROOT file")
    ap.add_argument("-O", "--opfilename", required=True, help="Output file stem")
    ap.add_argument("-s", "--start", type=int, default=0, help="Start entry for the PU file (default: 0)")
    ap.add_argument("-e", "--end", type=int, required=True, help="End entry for the PU file")
    ap.add_argument("-t", "--tree", default="hcalIsoTrkAnalyzer/CalibTree", help="Tree path inside the ROOT files")
    ap.add_argument("--max-depth", type=int, default=7, help="Highest HCAL depth to bin separately (default: 7)")
    ap.add_argument("--scale", type=int, default=5_000_000, help="No-PU chunk size per iteration (default: 5e6)")
    args = ap.parse_args()

    print("loading files")
    treePU = uproot.open(args.filePU)[args.tree]
    treeNPU = uproot.open(args.fileNPU)[args.tree]
    print("loaded files")

    dfspu = read_with_depth(treePU, PU_CONTROL_BRANCHES, args.max_depth,
                             entry_start=args.start, entry_stop=args.end)
    print("PU sample size:", dfspu.shape[0])

    npu_entries = treeNPU.num_entries
    npu_start = 0
    i = 0

    while npu_start < npu_entries:
        npu_stop = min(npu_start + args.scale, npu_entries)
        dfsnpu = read_with_depth(treeNPU, NOPU_BRANCHES, args.max_depth,
                                  entry_start=npu_start, entry_stop=npu_stop)
        print(f"loaded % of no-PU file = {100.0 * npu_stop / npu_entries:.2f}")
        print("no-PU chunk size:", dfsnpu.shape[0])

        dfspu_cols, dfsnpu_cols = set(dfspu.columns), set(dfsnpu.columns)
        merged = pd.merge(dfspu, dfsnpu, on=MERGE_KEYS)
        print("selected common events before cut:", merged.shape[0])

        if merged.empty:
            print("no common events in this chunk; skipping selections.")
            npu_start = npu_stop
            i += 1
            continue

        eHcal_pu = pu_colname("t_eHcal", dfspu_cols, dfsnpu_cols)
        eHcal_npu = npu_colname("t_eHcal", dfspu_cols, dfsnpu_cols)
        if eHcal_npu is None:
            print("WARNING: 't_eHcal' not found in the no-PU tree -- "
                  "no no-PU truth energy column will be in the output.")

        eHcal10_all = pu_colname("t_eHcal10_depthAll", dfspu_cols, dfsnpu_cols)
        eHcal30_all = pu_colname("t_eHcal30_depthAll", dfspu_cols, dfsnpu_cols)

        base_all = (
            merged["t_selectTk"] & merged["t_qltyFlag"]
            & (merged["t_hmaxNearP"] < 20) & (merged["t_eMipDR"] < 1)
            & ((merged["t_p"] - 50).abs() < 10) & (merged[eHcal_pu] > 10)
        )
        selections = {
            "all": base_all,
            "lo": base_all & (merged["t_ieta"].abs() < 16),
            "hi": base_all & (merged["t_ieta"].abs() > 15),
        }

        depth_cols = sorted(c for c in merged.columns if "_depth" in c)
        base_keepvars = [c for c in (
            "t_nVtx", "t_ieta", "t_hmaxNearP", "t_emaxNearP", "t_hAnnular",
            "t_eAnnular", "t_rhoh", "t_pt", "t_p", "t_eMipDR", "t_EventWeight",
            eHcal_pu, eHcal_npu,
        ) if c is not None and c in merged.columns]
        keepvars = base_keepvars + depth_cols

        # total-level delta (PU outer cone minus inner cone)
        if eHcal10_all in merged.columns and eHcal30_all in merged.columns:
            merged["t_delta"] = merged[eHcal30_all] - merged[eHcal10_all]
            keepvars.append("t_delta")

        # per-depth analog of t_delta, using the PU-side depth columns
        for d in range(1, args.max_depth + 1):
            col30 = pu_colname(f"t_eHcal30_depth{d}", dfspu_cols, dfsnpu_cols)
            col10 = pu_colname(f"t_eHcal10_depth{d}", dfspu_cols, dfsnpu_cols)
            if col30 in merged.columns and col10 in merged.columns:
                dcol = f"t_delta_depth{d}"
                merged[dcol] = merged[col30] - merged[col10]
                keepvars.append(dcol)

        for tag, sel in selections.items():
            sub = merged.loc[sel].reset_index(drop=True)
            print(f"selected events after cut [{tag}]:", sub.shape[0])

            if sub.empty:
                continue

            out_df = sub[[c for c in keepvars if c in sub.columns]]
            stem = f"{args.opfilename}_{i}_{args.start}_{args.end}_{tag}"
            out_df.to_parquet(stem + ".parquet")
            out_df.to_csv(stem + ".txt")

        npu_start = npu_stop
        i += 1


if __name__ == "__main__":
    main()
