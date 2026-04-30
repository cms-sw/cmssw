#!/bin/bash -e
# Smoke test for the Angantyr (HeavyIon:mode=2) code path in the Pythia8
# hadronizers, running through Pythia8ConcurrentGeneratorFilter.
#
# Drives two example fragments from Configuration/Generator:
#   - Pythia8Angantyr_pO_9p6TeV_MinimumBias_cfi.py
#                       : AngantyrInitialState = cms.PSet()   (refit every LS)
#   - Pythia8Angantyr_pO_9p6TeV_MinimumBias_skipRefit_cfi.py
#                       : AngantyrInitialState = cms.PSet(
#                             skipRefit = cms.untracked.bool(True) )
#
# Each fragment runs 200 events with numberEventsInLuminosityBlock=100 so the
# job crosses an LS 1 -> LS 2 boundary, first single-threaded and then with
# four concurrent streams.
#
# Expected (as of Pythia8 8.311 / 8.316):
#   * skipRefit=true  : runs clean, 200 events, exit 0.
#   * refit (default) : crashes at event 101 with
#                       vector::_M_range_check: __n (which is 0) >= this->size() (which is 0)
#                       from inside Pythia8. This is a known upstream bug
#                       exposed by successive Pythia::init() calls on an
#                       Angantyr-configured master generator. The refit cases
#                       are marked with `|| true` and only reported; they do
#                       not fail the test.
#
# The test fails (non-zero exit) if either skipRefit run does not finish cleanly.

declare -i FAILED=0

run_skiprefit() {
  local nthreads=$1
  local tag="HIN_pO_Angantyr_skipRefit_${nthreads}th"
  echo "================  ${tag}  ================"
  cmsDriver.py Configuration/Generator/python/Pythia8Angantyr_pO_9p6TeV_MinimumBias_skipRefit_cfi.py \
    --python_filename "${tag}_cfg.py" \
    --eventcontent RAWSIM --datatier GEN \
    --fileout "file:${tag}.root" \
    --conditions auto:phase1_2025_realistic \
    --beamspot Realistic25ns13TeVEarly2017Collision \
    --customise_commands "process.source.numberEventsInLuminosityBlock=cms.untracked.uint32(100)" \
    --step GEN --geometry DB:Extended --era Run3_2025_UPC_OXY \
    --mc -n 200 --nThreads "${nthreads}" --nConcurrentLumis 1 --no_exec
  if cmsRun "${tag}_cfg.py"; then
    echo "PASS: ${tag}"
  else
    echo "FAIL: ${tag}"
    FAILED+=1
  fi
}

run_refit_expect_crash() {
  local nthreads=$1
  local tag="HIN_pO_Angantyr_refit_${nthreads}th"
  echo "================  ${tag} (known Pythia8 bug; non-fatal to this test)  ================"
  cmsDriver.py Configuration/Generator/python/Pythia8Angantyr_pO_9p6TeV_MinimumBias_cfi.py \
    --python_filename "${tag}_cfg.py" \
    --eventcontent RAWSIM --datatier GEN \
    --fileout "file:${tag}.root" \
    --conditions auto:phase1_2025_realistic \
    --beamspot Realistic25ns13TeVEarly2017Collision \
    --customise_commands "process.source.numberEventsInLuminosityBlock=cms.untracked.uint32(100)" \
    --step GEN --geometry DB:Extended --era Run3_2025_UPC_OXY \
    --mc -n 200 --nThreads "${nthreads}" --nConcurrentLumis 1 --no_exec
  if cmsRun "${tag}_cfg.py"; then
    echo "UNEXPECTED PASS: ${tag} — Pythia8 Angantyr re-init bug may be fixed upstream; " \
         "consider dropping the skipRefit workaround."
  else
    echo "EXPECTED FAIL: ${tag} — Pythia8 Angantyr re-init crash reproduced."
  fi
}

run_skiprefit 1
run_skiprefit 4
run_refit_expect_crash 1 || true
run_refit_expect_crash 4 || true

if (( FAILED > 0 )); then
  echo "Summary: ${FAILED} skipRefit run(s) failed."
  exit 1
fi
echo "Summary: all skipRefit runs passed."
