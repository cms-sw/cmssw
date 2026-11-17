#!/bin/bash
# Generation, chain and tool checks on the offline Pythia sample (all in the test's working directory; the ROOT files are removed at the end).
# The NanoAOD tables of the same chain are tested in PhysicsTools/NanoAOD/test (testPatGenJetTables).
set -e
T=${LOCALTOP}/src/PhysicsTools/PatAlgos/test
echo "== generation: 20 Pythia ttbar events, gen jets, pruned/packed gen particles"
cmsRun $T/patGenJet_gen_ttbar20_cfg.py output=ttbar20_gen.root > gen.log 2>&1 || { tail -30 gen.log; exit 1; }
echo "== pat::GenJet chain and addPatGenJetCollection give the same collections"
cmsRun $T/patGenJet_test_tool_cfg.py input=ttbar20_gen.root output=tool.root > tool.log 2>&1 || { tail -40 tool.log; exit 1; }
python3 $T/patGenJet_check_tool_output.py tool.root
echo "== patGenJetsAK4 with embedGenCandidates and the forward maps prunedGenParticlesWithStatusOne, prunedGenParticles"
cmsRun $T/patGenJet_test_tool_cfg.py input=ttbar20_gen.root output=embed.root embed=True > embed.log 2>&1 || { tail -40 embed.log; exit 1; }
python3 $T/patGenJet_embed_stats.py tool.root embed.root
python3 $T/patGenJet_fwlite_read.py tool.root --label packedPatGenJetsAK8 --max-events 2 > fwlite.log 2>&1 || { tail -20 fwlite.log; exit 1; }
tail -1 fwlite.log
rm -f ttbar20_gen.root tool.root embed.root
echo "chain test OK"
