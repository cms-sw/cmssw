#!/bin/bash
#
# Unit test for the Phase-2 HLT menu tools (HLTrigger/Configuration/scripts):
#   1. hltPhase2MenuCheck : static checks of python/HLT_75e33 (fails on errors)
#   2. cmsDriver.py       : builds a configuration with -s L1P2GT,HLT:75e33 (--no_exec)
#   3. hltPhase2MenuDump  : loads it, writes JSON + HTML, lists seeds, sanity-checks the model
#   4. same for the alpaka variant, plus a diff against the default menu
#
# Run with:  scram b runtests_testHLTPhase2MenuTools
# Conditions/geometry/era default to the values of the Phase-2 HLT recipes and can be
# overridden through the environment (keep them in sync with the Phase-2 relval workflows).

set -o pipefail

: "${P2_CONDITIONS:=auto:phase2_realistic_T35}"
: "${P2_GEOMETRY:=ExtendedRun4D121}"
: "${P2_ERA:=Phase2C22I13M9}"
: "${P2_MIN_HLT_PATHS:=10}"

function die { echo "FAILED: $1 (status $2)"; exit "${2:-1}"; }

WORKDIR=$(mktemp -d -t hltPhase2MenuTools.XXXXXX)
echo $WORKDIR
trap 'rm -rf "${WORKDIR}"' EXIT
cd "${WORKDIR}" || die "cannot enter ${WORKDIR}" 1

echo ">>> 1. static checks of HLT_75e33"
hltPhase2MenuCheck --json check.json || die "hltPhase2MenuCheck reported errors" $?

function make_cfg {   # $1 = name, rest = extra cmsDriver options
  local name=$1; shift
  cmsDriver.py "${name}" -s L1P2GT,HLT:75e33 --processName=HLTX \
    --conditions "${P2_CONDITIONS}" --geometry "${P2_GEOMETRY}" --era "${P2_ERA}" \
    --eventcontent FEVTDEBUGHLT --filein=file:dummy.root --mc -n 1 \
    --python_filename="${name}_cfg.py" --no_exec "$@" > "${name}_driver.log" 2>&1 \
    || { cat "${name}_driver.log"; die "cmsDriver.py for ${name}" $?; }
}

echo ">>> 2. building configurations"
make_cfg Phase2
make_cfg Phase2_Alpaka --procModifiers alpaka

echo ">>> 3. dumping the default menu"
hltPhase2MenuDump Phase2_cfg.py --hlt-only --json default.json -o html_default || die "hltPhase2MenuDump (default)" $?
hltPhase2MenuDump default.json --list seeds > seeds.txt || die "hltPhase2MenuDump --list seeds" $?
hltPhase2MenuDump default.json --list eventcontent > eventcontent.txt || die "hltPhase2MenuDump --list eventcontent" $?

python3 - "${P2_MIN_HLT_PATHS}" <<'EOF' || die "menu model sanity checks" $?
import json, sys
m = json.load(open("default.json"))
hlt = [n for n, p in m["paths"].items() if p["kind"] == "path" and n.startswith("HLT_")]
problems = []
if len(hlt) < int(sys.argv[1]):
    problems.append("only %d HLT_ paths found" % len(hlt))
empty = [n for n in hlt if not m["paths"][n]["modules"]]
if empty:
    problems.append("paths with no modules: %s" % ", ".join(empty))
for n in hlt:
    for e in m["paths"][n]["modules"]:
        if e["label"] not in m["modules"]:
            problems.append("%s references unknown module %s" % (n, e["label"]))
print("%d HLT paths, %d modules, %d paths without L1 seed, %d modules in no path"
      % (len(hlt), len(m["modules"]), sum(1 for n in hlt if not m["paths"][n]["seed"]), len(m["unused_modules"])))
for p in problems:
    print("ERROR: " + p)
sys.exit(1 if problems else 0)
EOF
test -s html_default/index.html || die "html_default/index.html not written" 1

echo ">>> 4. alpaka menu and diff against default"
hltPhase2MenuDump Phase2_Alpaka_cfg.py --hlt-only --compare default.json -o html_alpaka --json alpaka.json \
  || die "hltPhase2MenuDump (alpaka)" $?
test -s html_alpaka/diff.html || die "html_alpaka/diff.html not written" 1

echo ">>> all Phase-2 menu tool tests passed"
