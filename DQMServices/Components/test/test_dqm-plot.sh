#!/bin/bash

function die { echo $1: status $2; exit $2; }

REMOTE="/store/group/phys_tracking/cmssw_unittests/"
DQMFILE="DQM_V0001_R000000001__RelValTTbar_14TeV__CMSSW_12_1_0_pre5-121X_mcRun3_2021_realistic_v15-v1__DQMIO.root"
SRC="DQMData/Run 1/HLT/Run summary/Tracking/ValidationWRTOffline/*"
ONE="DQMData/Run 1/HLT/Run summary/Tracking/ValidationWRTOffline/hltMergedWrtHighPurity/Eff_eta"

# Run dqm-plot, failing on a non-zero exit code or on any silently-failed
# plotting task (dqm-plot keeps going and only prints "N tasks failed ...").
function run_plot {
    desc="$1"; shift
    dqm-plot "$@" > dqm-plot.log 2>&1
    rc=$?
    if [ $rc -ne 0 ]; then cat dqm-plot.log; die "failed running dqm-plot (${desc})" $rc; fi
    if grep -q "tasks failed out of" dqm-plot.log; then
        cat dqm-plot.log; die "dqm-plot reported failed plotting tasks (${desc})" 1
    fi
}

# Tracking resolution / pull plots label check. Input collections in the input file have
# no "Sigma"/"Mean" plots, whose titles mix hand-built mathtext with ROOT #-notation
# (e.g. "#sigma(cot(#theta)) vs #eta Sigma"); those axis labels are reconstructed
# from the plot title and must render as valid matplotlib mathtext. Exercise that
# path directly so a broken conversion is caught.
python3 - <<'PYEOF'
import importlib.machinery, importlib.util, shutil, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import ROOT
ROOT.gErrorIgnoreLevel = ROOT.kError
loader = importlib.machinery.SourceFileLoader("dqmplot", shutil.which("dqm-plot"))
m = importlib.util.module_from_spec(importlib.util.spec_from_loader("dqmplot", loader))
loader.exec_module(m)
plotter = m.DQMPlotter()
titles = [
    "#sigma(cot(#theta)) vs #eta Sigma",
    "#sigma(cot(#theta)) vs #eta Mean",
    "#sigma(#phi) vs #eta Sigma",
    "normalized #chi^{2}",
    "Charge MisID Rate vs #phi",
]
for t in titles:
    h = ROOT.TH1F("h", "", 10, 0, 1); h.SetTitle(t); h.SetDirectory(0)
    for label in plotter.extract_labels_from_hist(h):
        fig = plt.figure(); fig.text(0.5, 0.5, label); fig.canvas.draw(); plt.close(fig)
print("label rendering check OK")
PYEOF
[ $? -eq 0 ] || die "axis-label mathtext rendering check failed" 1

# Unit checks for the --overlay comma-separated file-index syntax. These run
# without the remote DQM file.
python3 - <<'PYEOF'
import importlib.machinery, importlib.util, shutil, matplotlib
matplotlib.use("Agg")
loader = importlib.machinery.SourceFileLoader("dqmplot", shutil.which("dqm-plot"))
m = importlib.util.module_from_spec(importlib.util.spec_from_loader("dqmplot", loader))
loader.exec_module(m)
p = m.DQMPlotter()

# --overlay: comma-separated file indices in a single @ selector.
# 'collA@1,2:collB@2,3' overlays collA from files 1 and 2 with collB from
# files 2 and 3, producing series in file-major order.
jobs = p._parse_overlay_groups(["collA@1,2:collB@2,3"], 3)
assert len(jobs) == 1, jobs
j = jobs[0]
assert j["patterns"] == ["collA", "collB"], j["patterns"]
assert j["file_patterns"][0] == ["collA"], j["file_patterns"][0]
assert j["file_patterns"][1] == ["collA", "collB"], j["file_patterns"][1]
assert j["file_patterns"][2] == ["collB"], j["file_patterns"][2]

# Backward compatibility: no @ selector overlays every file.
jobs = p._parse_overlay_groups(["collA:collB"], 2)
assert jobs[0]["file_patterns"] == [["collA", "collB"], ["collA", "collB"]], jobs[0]

# A single 1-based index still resolves to that one file only.
jobs = p._parse_overlay_groups(["collA@1:collB@2"], 2)
assert jobs[0]["file_patterns"] == [["collA"], ["collB"]], jobs[0]

# Out-of-range indices must be rejected.
try:
    p._parse_overlay_groups(["collA@1,9:collB@2"], 2)
    raise AssertionError("out-of-range file index not rejected")
except ValueError as e:
    assert "out of range" in str(e), str(e)

# Non-numeric selectors must be rejected.
try:
    p._parse_overlay_groups(["collA@x"], 2)
    raise AssertionError("invalid file selector not rejected")
except ValueError as e:
    assert "invalid file selector" in str(e), str(e)

print("overlay unit checks OK")
PYEOF
[ $? -eq 0 ] || die "overlay unit checks failed" 1

COMMMAND=`xrdfs cms-xrd-global.cern.ch locate ${REMOTE}${DQMFILE}`
STATUS=$?
echo "xrdfs command status = "$STATUS

if [ $STATUS -eq 0 ]; then
    echo "Using file ${DQMFILE}. Running in ${LOCAL_TEST_DIR}."
    xrdcp root://cms-xrd-global.cern.ch/${REMOTE}${DQMFILE} .

    #  web + pdf + simple overlay + energy text + comma legend.
    run_plot "overlay + web + pdf" \
        -n 4 --web --pdf -s "${SRC}" \
        --overlay "hltMergedWrtHighPurity:hltMergedWrtHighPurityPV" \
        --energy-text "TEST" -l "File 1,File 2" -o plots \
        ./${DQMFILE} ./${DQMFILE}

    # Regression: omitting -l/--legend must not crash (default filename labels).
    run_plot "no --legend" \
        -n 4 -s "${SRC}" \
        --overlay "hltMergedWrtHighPurity:hltMergedWrtHighPurityPV" \
        -o plots_no_legend ./${DQMFILE} ./${DQMFILE}

    # Per-file --overlay syntax (collection@file) and per-job output folder.
    run_plot "per-file --overlay" \
        -n 4 -s "${SRC}" \
        --overlay "hltMergedWrtHighPurity@1:hltMergedWrtHighPurityPV@2" \
        -l "File 1,File 2" -o plots_overlay_perfile ./${DQMFILE} ./${DQMFILE}
    ls plots_overlay_perfile/overlay/hltMergedWrtHighPurity+hltMergedWrtHighPurityPV/*.png >/dev/null 2>&1 \
        || die 'per-file --overlay did not create overlay/<collections>/ plots' 1

    # New --overlay comma-separated file-index syntax: bind one collection to
    # a comma-separated list of 1-based file indices within a single @selector.
    # '...@1,2' overlays that collection from both files at once, equivalent to
    # the backward-compatible form without an @ selector here (two files).
    run_plot "comma-index --overlay" \
        -n 4 -s "${SRC}" \
        --overlay "hltMergedWrtHighPurity@1,2:hltMergedWrtHighPurityPV@1,2" \
        -l "File 1,File 2" -o plots_overlay_comma ./${DQMFILE} ./${DQMFILE}
    ls plots_overlay_comma/overlay/hltMergedWrtHighPurity+hltMergedWrtHighPurityPV/*.png >/dev/null 2>&1 \
        || die 'comma-index --overlay did not create overlay/<collections>/ plots' 1

    # Out-of-range comma index must be rejected with a non-zero exit code.
    dqm-plot -n 4 -s "${SRC}" \
        --overlay "hltMergedWrtHighPurity@1,9:hltMergedWrtHighPurityPV@2" \
        -l "File 1,File 2" -o plots_overlay_badidx ./${DQMFILE} ./${DQMFILE} \
        > dqm-plot-badidx.log 2>&1
    [ $? -ne 0 ] && grep -q "out of range" dqm-plot-badidx.log \
        || die 'out-of-range comma file index was not rejected' 1

    # --overlay-legend / --overlay-legend-title customise the overlay legend.
    run_plot "--overlay-legend" \
        -n 4 -s "${SRC}" \
        --overlay "hltMergedWrtHighPurity@1:hltMergedWrtHighPurityPV@2" \
        --overlay-legend "Merged HP,Merged HP (PV)" \
        --overlay-legend-title "Tracking collections" \
        -l "File 1,File 2" -o plots_overlay_legend ./${DQMFILE} ./${DQMFILE}

    # Plot-style options exercised together on a single histogram (fast).
    run_plot "style options" \
        -n 4 -s "${ONE}" \
        --logy --normalize --no-ratio --no-grid --rebin 2 \
        --title "Title" --xtitle "x" --ytitle "y" --legend-title "Legend" \
        -l "File 1,File 2" -o plots_style ./${DQMFILE} ./${DQMFILE}

    # --complement (purity -> fake rate) on a single histogram.
    run_plot "--complement" \
        -n 4 -s "${ONE}" --complement \
        -l "File 1,File 2" -o plots_complement ./${DQMFILE} ./${DQMFILE}

    # --ratio-label: auto ("Ratio wrt <first legend label>") and custom label.
    run_plot "--ratio-label auto" \
        -n 4 -s "${ONE}" --ratio-label auto \
        -l "File 1,File 2" -o plots_ratio_auto ./${DQMFILE} ./${DQMFILE}
    run_plot "--ratio-label custom" \
        -n 4 -s "${ONE}" --ratio-label "Custom ratio" \
        -l "File 1,File 2" -o plots_ratio_custom ./${DQMFILE} ./${DQMFILE}

    rm -fr ./${DQMFILE}
else
  die "SKIPPING test, file ${DQMFILE} not found" 0
fi
