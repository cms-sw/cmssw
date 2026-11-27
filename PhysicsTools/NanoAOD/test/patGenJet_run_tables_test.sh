#!/bin/bash
# Check on the offline Pythia sample: pat::GenJet chain and the three NanoAOD tables next to today's gen tables
# (all in the test's working directory; the ROOT files are removed at the end)
set -e
T=${LOCALTOP}/src/PhysicsTools/NanoAOD/test
GEN=$(edmFileInPath PhysicsTools/PatAlgos/test/patGenJet_gen_ttbar20_cfg.py)
echo "== generation: 20 Pythia ttbar events, gen jets, pruned/packed gen particles"
cmsRun $GEN output=ttbar20_gen.root > gen.log 2>&1 || { tail -30 gen.log; exit 1; }
echo "== pat::GenJet chain and the three NanoAOD tables next to today's gen tables"
cmsRun $T/run_patgenjets_cfg.py input=ttbar20_gen.root output=ttbar20_nano.root > run.log 2>&1 || { tail -40 run.log; exit 1; }
python3 $T/patGenJet_compare_gen_tables.py ttbar20_nano.root
rm -f ttbar20_gen.root ttbar20_nano.root
echo "tables test OK"
