#!/usr/bin/env python3
"""
L1 Track CI Comparison Unit Test

Compares the PR's tracking performance against two references, metric by metric:
  1. The IB (current code): catches a sudden degradation introduced by this PR.
  2. The long-term baseline (table at the bottom of this file): catches slow drift
     that stays within the IB tolerance on every single PR.
Passes only if no metric is worse than its allowed limit in either check.

COMMAND-LINE ARGS:
  -i <IB results dir>   output dir of an L1TKRunTest.sh run on the IB
  -p <PR results dir>   output dir of an L1TKRunTest.sh run on the PR
  Each dir is expected to look like (only results.out is read; anything
  else under <ALGO>/ is ignored):
      <dir>/HYBRID/results.out
      <dir>/HYBRID_NEWKF/results.out
      <dir>/HYBRID_DISPLACED/results.out

NOTE: results.out is written by the plotting macro
  L1Trigger/TrackFindingTracklet/test/L1TrackNtuplePlot.C (run via makeHists.csh).
  Its line wording dictates the METRICS text below -- if the macro's output
  changes, update METRICS to match (a missing line is reported as a FAILURE).

Event counts (fewer events gives too much statistical noise): the IB reference
needs at least BASELINE_EVENTS -- the event count the long-term baseline was produced
with (set next to BASELINE_REFERENCE at the bottom of this file) -- and the PR needs
at least as many events as the IB. Together this guarantees the PR has at least
as many events as both references. More is fine.
"""

import argparse
import os
import re
import sys

N_CHECKS = 0  # number of metric checks performed, for the summary line

def report(ok, message):
    """Count every check; print only failed ones, to keep the log short on success."""
    global N_CHECKS
    N_CHECKS += 1
    if not ok:
        print(f"FAILURE -- {message}")

ALGOS = ["HYBRID", "HYBRID_NEWKF", "HYBRID_DISPLACED"]

# TODO: replace both with the error-based threshold calculation.
# Placeholders: allow the PR to be up to this many % worse than the reference.
TOLERANCE_PCT = 2          # vs the IB
BASELINE_TOLERANCE_PCT = 5 # vs the long-term baseline (wider: it covers a long stretch of time)

# =============================================================================
# METRICS -- one entry per checked value: (name, exact text of its line in results.out, mode)
#   "min" = higher is better (fails if PR drops below the limit)
#   "max" = lower is better  (fails if PR rises above the limit)
# The name is also the key used in the BASELINE_REFERENCE table.
# =============================================================================
METRICS = [
    ("effi |eta|<1.0",        "efficiency for |eta| < 1.0 =",        "min"),
    ("effi 1.0<|eta|<1.75",   "efficiency for 1.0 < |eta| < 1.75 =", "min"),
    ("effi 1.75<|eta|<max",   "efficiency for 1.75 < |eta| <",       "min"),  # upper edge = the eta cut
    ("effi pt>8.0",           "efficiency for pt > 8.0 =",           "min"),
    ("effi pt>40.0",          "efficiency for pt > 40.0 =",          "min"),
    ("TP/event pt>3.0",       "TP/event (pt > 3.0) =",               "min"),
    ("TP/event pt>10.0",      "TP/event (pt > 10.0) =",              "min"),
    ("tracks/event pt>3.0",   "tracks/event (pt > 3.0) =",           "max"),
    ("tracks/event pt>10.0",  "tracks/event (pt > 10.0) =",          "max"),
    ("fake tracks %",         "Percentage fake tracks",              "max"),
    ("duplicate tracks %",    "Percentage duplicate tracks",         "max"),
    ("z0 res |eta|=0.05",     "at |eta| = 0.05",                     "max"),
    ("z0 res |eta|=1.95",     "at |eta| = 1.95",                     "max"),
]


# =============================================================================
# HELPERS
# =============================================================================

class MissingValue(Exception):
    """A metric's line was not found in results.out, or held no number."""


# First number after the first "=", which covers every line format:
#   "... = 95.49 +- 1.80"   "... = 3.23% 1796 1856"   "... = 0.10cm at |eta| = 0.05"
NUMBER_AFTER_EQUALS = re.compile(r"^[^=]*=\s*([0-9]+(?:\.[0-9]+)?)")


def get_value(lines, text, path):
    """Return the value on the first line containing `text` (ignoring capitalisation),
    as a string (kept for printing exactly as written in results.out).
    Raises MissingValue if absent."""
    for line in lines:
        if text.lower() in line.lower():
            match = NUMBER_AFTER_EQUALS.match(line)
            if match:
                return match.group(1)
            break
    raise MissingValue(f"'{text}' not found or not numeric in {path}")


def limit_for(reference, mode, tolerance_pct):
    """Allowed limit derived from a reference value, rounded to 4 decimals."""
    if mode == "min":
        return round(reference * (1 - tolerance_pct / 100), 4)
    return round(reference * (1 + tolerance_pct / 100), 4)


def is_worse(value, limit, mode):
    """True if value is on the wrong side of limit for this metric's direction."""
    return value < limit if mode == "min" else value > limit


def read_lines(path):
    with open(path) as f:
        return f.read().splitlines()


def check_baseline_reference():
    """Make sure BASELINE_REFERENCE has a value for every algorithm and every metric,
    and no unknown (e.g. misspelled) metric names. Returns a list of problems."""
    problems = []
    metric_names = [name for name, _, _ in METRICS]
    for algo in ALGOS:
        if algo not in BASELINE_REFERENCE:
            problems.append(f"no baseline reference defined for {algo}")
            continue
        table = BASELINE_REFERENCE[algo]
        for name in metric_names:
            if name not in table:
                problems.append(f"[{algo}] missing baseline reference value for '{name}'")
        for name in table:
            if name not in metric_names:
                problems.append(f"[{algo}] unknown metric '{name}' (not in METRICS -- typo?)")
    return problems


def check_algo(algo, ib_dir, pr_dir):
    """Compare every metric for one algorithm. Returns True if all checks passed."""
    ib_file = os.path.join(ib_dir, algo, "results.out")
    pr_file = os.path.join(pr_dir, algo, "results.out")

    # Both results.out files must exist
    for path in (ib_file, pr_file):
        if not os.path.isfile(path):
            print(f"FAILURE -- {path} not found")
            return False

    ib_lines = read_lines(ib_file)
    pr_lines = read_lines(pr_file)

    try:
        # Event counts: the IB needs at least BASELINE_EVENTS, and the PR at least as
        # many as the IB -- so the PR always has >= the events of both references.
        ib_events = int(float(get_value(ib_lines, "Number of events =", ib_file)))
        pr_events = int(float(get_value(pr_lines, "Number of events =", pr_file)))
        if ib_events < BASELINE_EVENTS:
            print(f"FAILURE -- [{algo}] IB reference has {ib_events} events, fewer than the long-term baseline's {BASELINE_EVENTS}")
            return False
        if pr_events < ib_events:
            print(f"FAILURE -- [{algo}] PR has {pr_events} events, fewer than the IB's {ib_events}")
            return False

        baseline = BASELINE_REFERENCE[algo]   # completeness already checked by check_baseline_reference()

        passed = True
        for name, text, mode in METRICS:
            ib = get_value(ib_lines, text, ib_file)
            pr = get_value(pr_lines, text, pr_file)

            # 1. Short-term: PR vs IB
            limit = limit_for(float(ib), mode, TOLERANCE_PCT)
            worse = is_worse(float(pr), limit, mode)
            passed &= not worse
            report(not worse, f"[{algo}] {name}: PR = {pr}, IB = {ib} ({mode} allowed: {limit:.4f})")

            # 2. Long-term: PR vs long-term baseline
            ref = baseline[name]
            limit = limit_for(ref, mode, BASELINE_TOLERANCE_PCT)
            worse = is_worse(float(pr), limit, mode)
            passed &= not worse
            report(not worse, f"[{algo}] (baseline) {name}: PR = {pr}, baseline ref = {ref} ({mode} allowed: {limit:.4f})")
        return passed

    except MissingValue as err:
        print(f"FAILURE -- [{algo}] {err}")
        return False


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Compare PR tracking performance against the IB and the long-term baseline.")
    parser.add_argument("-i", dest="ib_dir", required=True, help="IB results dir (reference: current code)")
    parser.add_argument("-p", dest="pr_dir", required=True, help="PR results dir (under test)")
    args = parser.parse_args()

    # Check that both results directories exist: the IB run (-i) and the PR run (-p)
    for label, flag, path in (("IB", "-i", args.ib_dir), ("PR", "-p", args.pr_dir)):
        if not os.path.isdir(path):
            print(f"ERROR: {label} results directory ({flag}) not found: '{path}'")
            parser.print_usage()
            sys.exit(1)

    # Every algorithm must have a complete long-term baseline -- nothing is ever skipped
    problems = check_baseline_reference()
    if problems:
        for problem in problems:
            print(f"ERROR: BASELINE_REFERENCE {problem}")
        print("=== OVERALL RESULT: FAILURE -- fix BASELINE_REFERENCE at the bottom of this file ===")
        sys.exit(1)

    all_passed = True
    for algo in ALGOS:
        all_passed &= check_algo(algo, args.ib_dir, args.pr_dir)
    if not all_passed:
        print(f"=== OVERALL RESULT: FAILURE ({N_CHECKS} checks) -- see FAILURE lines above ===")
        sys.exit(1)
    print(f"=== OVERALL RESULT: SUCCESS ({N_CHECKS} checks) ===")


# =============================================================================
# LONG-TERM BASELINE -- long-term reference values, one dictionary per algorithm,
# keyed by the metric names in METRICS. Every algorithm in ALGOS must list every
# metric in METRICS, spelled exactly the same, with a value -- the test fails otherwise.
# Last updated: November 2026.
# Produced by: L1TKRunTest, 1000 events per algorithm, CMSSW_20_1_0_pre5
#              (central code, before the next L1TK integration -- refresh after it).
#              Displaced algorithms use the displaced cuts (pt > 3, |eta| < 2.0).
# Samples:     RelValTTbar_14TeV / RelValDisplacedSUSY_14TeV,
#              CMSSW_20_0_0_pre1-PU_150X_mcRun4_realistic_v1_STD_D121_RegeneratedGS_PU_16Aug26-v3
# =============================================================================
BASELINE_EVENTS = 1000   # events per algorithm used to produce BASELINE_REFERENCE (update with it).
                         # Also the minimum event count for the IB, and therefore the PR.

BASELINE_REFERENCE = {
    "HYBRID": {                 # ttbar+PU200
        "effi |eta|<1.0":        94.01,
        "effi 1.0<|eta|<1.75":   93.52,
        "effi 1.75<|eta|<max":   94.70,
        "effi pt>8.0":           94.04,
        "effi pt>40.0":          93.25,
        "TP/event pt>3.0":       50.75,
        "TP/event pt>10.0":      4.63,
        "tracks/event pt>3.0":   56.72,
        "tracks/event pt>10.0":  5.84,
        "fake tracks %":         2.57,
        "duplicate tracks %":    1.31,
        "z0 res |eta|=0.05":     0.10,
        "z0 res |eta|=1.95":     0.38,
    },
    "HYBRID_NEWKF": {           # ttbar+PU200
        "effi |eta|<1.0":        95.08,
        "effi 1.0<|eta|<1.75":   94.40,
        "effi 1.75<|eta|<max":   96.16,
        "effi pt>8.0":           94.87,
        "effi pt>40.0":          93.00,
        "TP/event pt>3.0":       50.75,
        "TP/event pt>10.0":      4.63,
        "tracks/event pt>3.0":   64.06,
        "tracks/event pt>10.0":  7.28,
        "fake tracks %":         5.13,
        "duplicate tracks %":    6.41,
        "z0 res |eta|=0.05":     0.12,
        "z0 res |eta|=1.95":     0.50,
    },
    "HYBRID_DISPLACED": {       # displacedSUSY+PU200, displaced cuts
        "effi |eta|<1.0":        85.86,
        "effi 1.0<|eta|<1.75":   85.38,
        "effi 1.75<|eta|<max":   81.78,
        "effi pt>8.0":           81.38,
        "effi pt>40.0":          74.07,
        "TP/event pt>3.0":       42.94,
        "TP/event pt>10.0":      4.95,
        "tracks/event pt>3.0":   155.87,
        "tracks/event pt>10.0":  45.44,
        "fake tracks %":         69.39,
        "duplicate tracks %":    1.25,
        "z0 res |eta|=0.05":     0.12,
        "z0 res |eta|=1.95":     0.38,
    },
}

# Run. Kept at the very end so everything above, including BASELINE_REFERENCE,
# is defined before main() starts.
if __name__ == "__main__":
    main()