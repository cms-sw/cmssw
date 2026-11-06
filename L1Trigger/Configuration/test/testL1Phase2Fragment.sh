#!/bin/bash
# drift test: L1_Phase2_cff must match the cmsDriver L1TrackTrigger,L1,L1P2GT configuration
python3 "${SCRAM_TEST_PATH}"/l1tPhase2Fragment.py validate || exit $?
