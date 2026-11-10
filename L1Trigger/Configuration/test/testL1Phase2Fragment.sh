#!/bin/bash
# drift test: L1_Phase2_cff must match the cmsDriver L1TrackTrigger,L1,L1P2GT configuration
python3 "${SCRAM_TEST_PATH}"/l1tPhase2Fragment.py validate || exit $?

# the L1P2 cmsDriver step (together with HLT, which also loads a fragment) must give a valid configuration
cmsDriver.py step2 -s DIGI:pdigi_valid,L1P2,DIGI2RAW,HLT:@relvalRun4 --conditions auto:phase2_realistic_T35 \
  --geometry ExtendedRun4D128 --era Phase2C26I13M9 --datatier GEN-SIM-DIGI-RAW --eventcontent FEVTDEBUGHLT \
  --filein file:step1.root --fileout file:step2.root -n 10 --mc --no_exec --python_filename l1p2_cfg.py || exit $?
python3 -c "
exec(open('l1p2_cfg.py').read())
from L1Trigger.Configuration.L1_Phase2_cff import l1Paths
paths = [p.label_() for p in process.schedule]
missing = [p for p in l1Paths if p not in paths]
assert not missing, 'L1 paths not in the schedule: %s' % missing
hltPaths = [i for i, p in enumerate(paths) if p.startswith('HLT_')]
assert hltPaths, 'no HLT path in the schedule'
assert max(paths.index(p) for p in l1Paths) < min(hltPaths), 'L1 paths must run before HLT'
print('L1P2 step: OK')
" || exit $?

# the L1P2 step on stored events: the MC-truth path is dropped, all other L1 paths are scheduled
cmsDriver.py l1 -s L1P2 --customise L1Trigger/Configuration/customisePhase2TTNoMC.customiseL1P2NoMCTruth \
  --conditions auto:phase2_realistic_T35 --geometry ExtendedRun4D128 --era Phase2C26I13M9 --datatier GEN-SIM-DIGI-RAW \
  --eventcontent FEVTDEBUGHLT --filein file:step2.root --fileout file:l1.root -n 10 --mc --no_exec \
  --processName L1P2 --python_filename l1p2nomc_cfg.py || exit $?
python3 -c "
exec(open('l1p2nomc_cfg.py').read())
from L1Trigger.Configuration.L1_Phase2_cff import l1Paths
paths = [p.label_() for p in process.schedule]
assert 'L1TrackTriggerMCTruth_step' not in paths, 'MC-truth path still scheduled'
missing = [p for p in l1Paths if p != 'L1TrackTriggerMCTruth_step' and p not in paths]
assert not missing, 'L1 paths not in the schedule: %s' % missing
print('L1P2 step without MC truth: OK')
" || exit $?
