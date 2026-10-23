#!/bin/bash

# Check that DST_NGTScouting accepts exactly the events accepted by hltL1GTAcceptFilter,
# i.e. that no other module in the path rejects events.
#  1. static check: the only EDFilters in DST_NGTScouting are hltL1GTAcceptFilter
#     and the pass-through ones (hltTriggerType always accepts MC, hltBoolEnd always accepts)
#  2. runtime check: run L1P2GT+HLT:NGTScouting with an extra path containing only
#     hltL1GTAcceptFilter, and compare the two TriggerResults bits event by event

# Pass in name and status
function die {
  printf "\n%s: status %s\n" "$1" "$2"
  if [ $# -gt 2 ]; then
    printf "%s\n" "=== Log File =========="
    cat $3
    printf "%s\n" "=== End of Log File ==="
  fi
  exit $2
}

INPUT_FILE="/store/relval/CMSSW_17_0_0_pre1/RelValTTbar_14TeV/GEN-SIM-DIGI-RAW/PU_150X_mcRun4_realistic_v1_STD_D121_ReHLT_PU-v1/2590000/027f0955-da87-419a-a537-6b6f93d54345.root"
EVENTS=${EVENTS:-10}
THREADS=${THREADS:-4}
PROCESS="HLTX"
CFG_FILE="NGTScouting_L1GTAccept_cfg.py"
OUTPUT_FILE="NGTScouting_L1GTAccept_TriggerResults.root"
LOG_FILE="NGTScouting_L1GTAccept.log"

# Step 1: Generate the configuration
echo "Running cmsDriver..."
cmsDriver.py NGTScouting \
  -s L1P2GT,HLT:NGTScouting \
  --processName=${PROCESS} \
  --conditions auto:phase2_realistic_T35 \
  --geometry ExtendedRun4D121 \
  --era Phase2C22I13M9 \
  --procModifiers alpaka,ngtScouting \
  --accelerators cpu \
  --filein="${INPUT_FILE}" \
  --mc \
  --nThreads ${THREADS} \
  --inputCommands 'keep *, drop *_hlt*_*_HLT, drop triggerTriggerFilterObjectWithRefs_l1t*_*_HLT' \
  -n ${EVENTS} \
  --no_output \
  --no_exec \
  --python_filename ${CFG_FILE} \
  || die 'Failure running cmsDriver' $?

# add a path with hltL1GTAcceptFilter alone, and keep only the TriggerResults of this process
cat << EOF >> "${CFG_FILE}"

process.NGTCheck_L1GTAccept = cms.Path(process.hltL1GTAcceptFilter)
process.schedule.append(process.NGTCheck_L1GTAccept)

process.ngtCheckOutput = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string("${OUTPUT_FILE}"),
    outputCommands = cms.untracked.vstring(
        "drop *",
        "keep edmTriggerResults_TriggerResults__${PROCESS}"
    )
)
process.ngtCheckOutputPath = cms.EndPath(process.ngtCheckOutput)
process.schedule.append(process.ngtCheckOutputPath)
EOF

# Step 2: Static check of the EDFilters in DST_PFScouting
echo "Checking the EDFilters in DST_PFScouting..."
python3 - "${CFG_FILE}" << 'EOF' || die 'Unexpected EDFilters in DST_PFScouting' $?
import sys
import FWCore.ParameterSet.Config as cms

namespace = {}
exec(open(sys.argv[1]).read(), namespace)
process = namespace['process']

allowed = {'hltTriggerType', 'hltL1GTAcceptFilter', 'hltBoolEnd', 'hltGoodOfflinePrimaryVertices'}
filters = sorted(label for label in process.DST_PFScouting.moduleNames()
                 if isinstance(getattr(process, label, None), cms.EDFilter))
unexpected = [label for label in filters if label not in allowed]
print('EDFilters in DST_PFScouting:', ', '.join(filters))
if unexpected:
    for label in unexpected:
        print(f'  unexpected EDFilter: {label} ({getattr(process, label).type_()})')
    sys.exit(1)
EOF

# Step 3: Run the job
echo "Running cmsRun..."
cmsRun "${CFG_FILE}" &> "${LOG_FILE}" \
  || die 'Failure running cmsRun' $? "${LOG_FILE}"

# Step 4: Compare DST_PFScouting and NGTCheck_L1GTAccept event by event
echo "Comparing the trigger results..."
python3 - "${OUTPUT_FILE}" "${PROCESS}" << 'EOF' || die 'DST_PFScouting differs from hltL1GTAcceptFilter' $?
import sys
import ROOT
from DataFormats.FWLite import Events, Handle

fileName, processName = sys.argv[1:3]
events = Events(fileName)
handle = Handle('edm::TriggerResults')

nEvents = nAccepted = 0
mismatches = []
for event in events:
    event.getByLabel(('TriggerResults', '', processName), handle)
    results = handle.product()
    names = event.object().triggerNames(results)
    dst = results.accept(names.triggerIndex('DST_PFScouting'))
    l1 = results.accept(names.triggerIndex('NGTCheck_L1GTAccept'))
    nEvents += 1
    nAccepted += l1
    if dst != l1:
        aux = event.eventAuxiliary()
        mismatches.append((aux.run(), aux.luminosityBlock(), aux.event(), dst, l1))

print(f'{nEvents} events, {nAccepted} accepted by hltL1GTAcceptFilter, {len(mismatches)} mismatches')
for run, lumi, evt, dst, l1 in mismatches:
    print(f'  {run}:{lumi}:{evt}  DST_PFScouting={dst}  hltL1GTAcceptFilter={l1}')
if nEvents == 0:
    print('no events processed')
    sys.exit(1)
if mismatches:
    sys.exit(1)
EOF

echo "DST_PFScouting matches hltL1GTAcceptFilter."
