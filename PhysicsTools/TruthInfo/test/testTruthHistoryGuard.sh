#!/bin/bash
# Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>
#
# History-guard integration test. Runs a short Phase-2 GEN,SIM job with the
# enableTruth process modifier (so g4SimHits reattaches dropped ancestors), builds
# the truth graph from the freshly simulated SimTracks/SimVertices and asserts -
# via TruthGraphTopologyChecker(failOnViolations=True) - that the SimTrack/SimVertex
# history is one tree fully connected to the generator. It FAILS if a simulation
# change drops the per-track parentage (the regression a GPU sim port can introduce).

function die { echo "$1: status $2" ; exit $2 ; }

cmsDriver.py SingleElectronPt35_pythia8_cfi \
  -s GEN,SIM -n 3 --nThreads 1 \
  --conditions auto:phase2_realistic_T35_13TeV \
  --beamspot DBrealisticHLLHC \
  --datatier GEN-SIM --eventcontent FEVTDEBUG \
  --geometry ExtendedRun4D120 --era Phase2C26I13M9 \
  --procModifiers enableTruth \
  --customise PhysicsTools/TruthInfo/addTruthHistoryGuard.addTruthHistoryGuard \
  --fileout file:testTruthHistoryGuard_genSim.root \
  --no_exec --python_filename testTruthHistoryGuard_genSim.py \
  || die "cmsDriver config generation" $?

cmsRun testTruthHistoryGuard_genSim.py || die "GEN-SIM truth-graph history guard" $?
