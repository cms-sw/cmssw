#!/bin/sh -ex

function die { echo $1: status $2 ;  exit $2; }

LOCAL_TEST_DIR=${SCRAM_TEST_PATH}

LD_PRELOAD="libPerfToolsAllocMonitorPreload.so" cmsRun ${LOCAL_TEST_DIR}/moduleAlloc_cfg.py || die 'Failure using moduleAlloc_cfg.py' $?

edmModuleAllocMonitorAnalyze.py -j moduleAlloc.log  > moduleAlloc.json
grep -A9 'cpptypes' moduleAlloc.json | sort --ignore-leading-blanks | grep -v 'cpptypes' | grep -v '}' | sed 's/,//g' > cpptypes.txt
diff --ignore-all-space cpptypes.txt ${LOCAL_TEST_DIR}/unittest_output/cpptypes.txt || die 'differences in edmModuleAllocMonitorAnalyzer.py output' $?

edmModuleAllocJsonToCircles.py moduleAlloc.json > moduleAlloc.circles.json
grep '"\(record\|type\|label\)": ".*",' moduleAlloc.circles.json > circles.txt
diff circles.txt ${LOCAL_TEST_DIR}/unittest_output/circles.txt || die 'differences in edmModuleAllocJsonToCircles.py output' $?

grep '^[fF]' moduleAlloc.log | awk '{print $1,$2,$3,$4,$5,$6}' > allTransitions.log
diff allTransitions.log ${LOCAL_TEST_DIR}/unittest_output/allTransitions.log || die 'differences in allTransitions' $?

grep '^[mM]' moduleAlloc.log | awk '{print $1,$2,$3,$4,$5}' > allEDModules.log
diff allEDModules.log ${LOCAL_TEST_DIR}/unittest_output/allEDModules.log || die 'differences in allEDModules' $?


grep '^[nN]' moduleAlloc.log | awk '{print $1,$2,$3,$4,$5,$6}' > allESModules.log
diff allESModules.log ${LOCAL_TEST_DIR}/unittest_output/allESModules.log || die 'differences in allESModules' $?


############### only 1 ED module kept
LD_PRELOAD="libPerfToolsAllocMonitorPreload.so" cmsRun ${LOCAL_TEST_DIR}/moduleAlloc_cfg.py --edmodule || die 'Failure using moduleAlloc_cfg.py --edmodule' $?

grep '^[mM]' moduleAlloc.log | awk '{print $1,$2,$3,$4,$5}' > only_ed_EDModules.log
diff only_ed_EDModules.log ${LOCAL_TEST_DIR}/unittest_output/only_ed_EDModules.log || die 'differences in only_ed_EDModules' $?


grep '^[nN]' moduleAlloc.log | awk '{print $1,$2,$3,$4,$5,$6}' > only_ed_ESModules.log
diff only_ed_ESModules.log ${LOCAL_TEST_DIR}/unittest_output/only_ed_ESModules.log || die 'differences in only_ed_ESModules' $?


############### only 1 ES module kept
LD_PRELOAD="libPerfToolsAllocMonitorPreload.so" cmsRun ${LOCAL_TEST_DIR}/moduleAlloc_cfg.py --esmodule || die 'Failure using moduleAlloc_cfg.py --esmodule' $?

grep '^[mM]' moduleAlloc.log | awk '{print $1,$2,$3,$4,$5}' > only_es_EDModules.log
diff only_es_EDModules.log ${LOCAL_TEST_DIR}/unittest_output/only_es_EDModules.log || die 'differences in only_es_EDModules' $?


grep '^[nN]' moduleAlloc.log | awk '{print $1,$2,$3,$4,$5,$6}' > only_es_ESModules.log
diff only_es_ESModules.log ${LOCAL_TEST_DIR}/unittest_output/only_es_ESModules.log || die 'differences in only_es_ESModules' $?

############## skip events
LD_PRELOAD="libPerfToolsAllocMonitorPreload.so" cmsRun ${LOCAL_TEST_DIR}/moduleAlloc_cfg.py --skipEvents || die 'Failure using moduleAlloc_cfg.py --skipEvents' $?

grep '^[fF]' moduleAlloc.log | awk '{print $1,$2,$3,$4,$5,$6}' > skipEvents_Transitions.log
diff skipEvents_Transitions.log ${LOCAL_TEST_DIR}/unittest_output/skipEvents_Transitions.log || die 'differences in skipEvents_Transitions' $?

grep '^[mM]' moduleAlloc.log | awk '{print $1,$2,$3,$4,$5}' > skipEvents_EDModules.log
diff skipEvents_EDModules.log ${LOCAL_TEST_DIR}/unittest_output/skipEvents_EDModules.log || die 'differences in skipEvents_EDModules' $?


grep '^[nN]' moduleAlloc.log | awk '{print $1,$2,$3,$4,$5,$6}' > skipEvents_ESModules.log
diff skipEvents_ESModules.log ${LOCAL_TEST_DIR}/unittest_output/skipEvents_ESModules.log || die 'differences in skipEvents_ESModules' $?
