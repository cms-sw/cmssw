#!/bin/sh -ex

function die { echo $1: status $2 ;  exit $2; }

LOCAL_TEST_DIR=${SCRAM_TEST_PATH}

LD_PRELOAD="libPerfToolsAllocMonitorPreload.so" cmsRun ${LOCAL_TEST_DIR}/moduleAlloc_cfg.py || die 'Failure using moduleAlloc_cfg.py' $?
mv moduleAlloc.log moduleAlloc.log.orig

edmModuleAllocMonitorAnalyze.py -j moduleAlloc.log.orig  > moduleAlloc.json
grep -A9 'cpptypes' moduleAlloc.json | sort --ignore-leading-blanks | grep -v 'cpptypes' | grep -v '}' | sed 's/,//g' > cpptypes.txt
diff --ignore-all-space cpptypes.txt ${LOCAL_TEST_DIR}/unittest_output/cpptypes.txt || die 'differences in edmModuleAllocMonitorAnalyzer.py output' $?

edmModuleAllocJsonToCircles.py moduleAlloc.json > moduleAlloc.circles.json
grep '"\(record\|type\|label\)": ".*",' moduleAlloc.circles.json > circles.txt
diff circles.txt ${LOCAL_TEST_DIR}/unittest_output/circles.txt || die 'differences in edmModuleAllocJsonToCircles.py output' $?

grep '^[fF]' moduleAlloc.log.orig | awk '{print $1,$2,$3,$4,$5,$6}' > allTransitions.log
diff allTransitions.log ${LOCAL_TEST_DIR}/unittest_output/allTransitions.log || die 'differences in allTransitions' $?

grep '^[mM]' moduleAlloc.log.orig | awk '{print $1,$2,$3,$4,$5}' > allEDModules.log
diff allEDModules.log ${LOCAL_TEST_DIR}/unittest_output/allEDModules.log || die 'differences in allEDModules' $?


grep '^[nN]' moduleAlloc.log.orig | awk '{print $1,$2,$3,$4,$5,$6}' > allESModules.log
diff allESModules.log ${LOCAL_TEST_DIR}/unittest_output/allESModules.log || die 'differences in allESModules' $?

# per-transition trace output
edmModuleAllocMonitorAnalyze.py --trace moduleAlloc.log.orig > moduleAlloc.trace.txt || die 'Failure using --trace' $?
grep -q 'starting action: .* during construction : id=' moduleAlloc.trace.txt || die 'trace missing construction action' $?
grep -q 'starting: begin job : id=' moduleAlloc.trace.txt || die 'trace missing begin job' $?
grep -q 'during event : id=' moduleAlloc.trace.txt || die 'trace missing event' $?

# per-module/per-transition/per-activity averaged summary
edmModuleAllocMonitorAnalyze.py moduleAlloc.log.orig > moduleAlloc.summary.txt || die 'Failure using default summary' $?
grep -q '^Module label source,'            moduleAlloc.summary.txt || die 'summary missing source' $?
grep -q '^Module label thingProducer,'     moduleAlloc.summary.txt || die 'summary missing thingProducer' $?
grep -q '^Module label Thing,'             moduleAlloc.summary.txt || die 'summary missing Thing' $?
grep -q '^Module label WhatsItESProducer,' moduleAlloc.summary.txt || die 'summary missing ES module' $?
grep -Eq '^ +event$'             moduleAlloc.summary.txt || die 'summary missing event transition' $?
grep -Eq '^ +process \(.*calls\)$' moduleAlloc.summary.txt || die 'summary missing calls note' $?
grep -Eq '^ +nAlloc '            moduleAlloc.summary.txt || die 'summary missing nAlloc avg' $?
grep -Eq '^ +nAlloc .* max1Alloc [0-9-]+$' moduleAlloc.summary.txt || die 'summary fields not on one row' $?
# EventSetup modules are split per record/callID
grep -Eq '^ +process record GadgetRcd callID 0 (.*)$' moduleAlloc.summary.txt || die 'summary missing ES record callID 0' $?
grep -Eq '^ +process record GadgetRcd callID 4 (.*)$' moduleAlloc.summary.txt || die 'summary missing ES record callID 4' $?

# --summaryField restricts the summary to a single quantity
edmModuleAllocMonitorAnalyze.py --summaryField max1Alloc moduleAlloc.log.orig > moduleAlloc.summary.max1Alloc.txt || die 'Failure using --summaryField max1Alloc' $?
grep -Eq '^ +max1Alloc [0-9-]+$' moduleAlloc.summary.max1Alloc.txt || die '--summaryField max1Alloc missing max1Alloc line' $?
grep -Eq '^ +nAlloc '           moduleAlloc.summary.max1Alloc.txt && die '--summaryField max1Alloc unexpectedly printed nAlloc' 1
grep -Eq '^ +nDealloc '         moduleAlloc.summary.max1Alloc.txt && die '--summaryField max1Alloc unexpectedly printed nDealloc' 1
grep -Eq '^ +added '            moduleAlloc.summary.max1Alloc.txt && die '--summaryField max1Alloc unexpectedly printed added' 1
grep -Eq '^ +minTemp '          moduleAlloc.summary.max1Alloc.txt && die '--summaryField max1Alloc unexpectedly printed minTemp' 1
grep -Eq '^ +maxTemp '          moduleAlloc.summary.max1Alloc.txt && die '--summaryField max1Alloc unexpectedly printed maxTemp' 1

# an unknown --summaryField value must be rejected by argparse
if edmModuleAllocMonitorAnalyze.py --summaryField bogus moduleAlloc.log.orig > /dev/null 2>summaryField_bad.log; then
  die '--summaryField bogus unexpectedly succeeded' 1
fi
grep -q 'invalid choice' summaryField_bad.log || die '--summaryField bogus did not report invalid choice' $?


############### only 1 ED module kept
LD_PRELOAD="libPerfToolsAllocMonitorPreload.so" cmsRun ${LOCAL_TEST_DIR}/moduleAlloc_cfg.py --edmodule || die 'Failure using moduleAlloc_cfg.py --edmodule' $?
mv moduleAlloc.log moduleAlloc.log.edmodule
grep '^[mM]' moduleAlloc.log.edmodule | awk '{print $1,$2,$3,$4,$5}' > only_ed_EDModules.log
diff only_ed_EDModules.log ${LOCAL_TEST_DIR}/unittest_output/only_ed_EDModules.log || die 'differences in only_ed_EDModules' $?



grep '^[nN]' moduleAlloc.log.edmodule | awk '{print $1,$2,$3,$4,$5,$6}' > only_ed_ESModules.log
diff only_ed_ESModules.log ${LOCAL_TEST_DIR}/unittest_output/only_ed_ESModules.log || die 'differences in only_ed_ESModules' $?

############### only 1 ES module kept
LD_PRELOAD="libPerfToolsAllocMonitorPreload.so" cmsRun ${LOCAL_TEST_DIR}/moduleAlloc_cfg.py --esmodule || die 'Failure using moduleAlloc_cfg.py --esmodule' $?
mv moduleAlloc.log moduleAlloc.log.esmodule
grep '^[mM]' moduleAlloc.log.esmodule | awk '{print $1,$2,$3,$4,$5}' > only_es_EDModules.log
diff only_es_EDModules.log ${LOCAL_TEST_DIR}/unittest_output/only_es_EDModules.log || die 'differences in only_es_EDModules' $?



grep '^[nN]' moduleAlloc.log.esmodule | awk '{print $1,$2,$3,$4,$5,$6}' > only_es_ESModules.log
diff only_es_ESModules.log ${LOCAL_TEST_DIR}/unittest_output/only_es_ESModules.log || die 'differences in only_es_ESModules' $?

############## skip events
LD_PRELOAD="libPerfToolsAllocMonitorPreload.so" cmsRun ${LOCAL_TEST_DIR}/moduleAlloc_cfg.py --skipEvents || die 'Failure using moduleAlloc_cfg.py --skipEvents' $?
mv moduleAlloc.log moduleAlloc.log.skipEvents
grep '^[fF]' moduleAlloc.log.skipEvents | awk '{print $1,$2,$3,$4,$5,$6}' > skipEvents_Transitions.log
diff skipEvents_Transitions.log ${LOCAL_TEST_DIR}/unittest_output/skipEvents_Transitions.log || die 'differences in skipEvents_Transitions' $?

grep '^[mM]' moduleAlloc.log.skipEvents | awk '{print $1,$2,$3,$4,$5}' > skipEvents_EDModules.log
diff skipEvents_EDModules.log ${LOCAL_TEST_DIR}/unittest_output/skipEvents_EDModules.log || die 'differences in skipEvents_EDModules' $?


grep '^[nN]' moduleAlloc.log.skipEvents | awk '{print $1,$2,$3,$4,$5,$6}' > skipEvents_ESModules.log
diff skipEvents_ESModules.log ${LOCAL_TEST_DIR}/unittest_output/skipEvents_ESModules.log || die 'differences in skipEvents_ESModules' $?

############### ExternalWork / Transformer / TransformAsync modules
LD_PRELOAD="libPerfToolsAllocMonitorPreload.so" cmsRun ${LOCAL_TEST_DIR}/moduleAllocAcquireTransform_cfg.py || die 'Failure using moduleAllocAcquireTransform_cfg.py' $?

grep '^[fF]' moduleAllocAcquireTransform.log | awk '{print $1,$2,$3,$4,$5,$6}' > acquireTransform_Transitions.log
diff acquireTransform_Transitions.log ${LOCAL_TEST_DIR}/unittest_output/acquireTransform_Transitions.log || die 'differences in acquireTransform_Transitions' $?

grep '^[mMaA]' moduleAllocAcquireTransform.log | awk '{print $1,$2,$3,$4,$5}' > acquireTransform_EDModules.log
diff acquireTransform_EDModules.log ${LOCAL_TEST_DIR}/unittest_output/acquireTransform_EDModules.log || die 'differences in acquireTransform_EDModules' $?

edmModuleAllocMonitorAnalyze.py -j moduleAllocAcquireTransform.log  > acquireTransform.json
grep -A7 'cpptypes' acquireTransform.json | sort --ignore-leading-blanks | grep -v 'cpptypes' | grep -v '}' | sed 's/,//g' > acquireTransform_cpptypes.txt
diff --ignore-all-space acquireTransform_cpptypes.txt ${LOCAL_TEST_DIR}/unittest_output/acquireTransform_cpptypes.txt || die 'differences in edmModuleAllocMonitorAnalyzer.py output for acquire+transform' $?

edmModuleAllocJsonToCircles.py acquireTransform.json > acquireTransform.circles.json
grep '"\(record\|type\|label\)": ".*",' acquireTransform.circles.json > acquireTransform.circles.txt
# to be enabled later when the circles output is stable
#diff acquireTransform.circles.txt ${LOCAL_TEST_DIR}/unittest_output/acquireTransform.circles.txt || die 'differences in edmModuleAllocJsonToCircles.py output' $?

# summary must expose non-process (acquire) activities on the event transition
edmModuleAllocMonitorAnalyze.py moduleAllocAcquireTransform.log > acquireTransform.summary.txt || die 'Failure using default summary for acquire+transform' $?
grep -q '^Module label externalWorkAllocProducer,'   acquireTransform.summary.txt || die 'summary missing externalWork module' $?
grep -q '^Module label transformAsyncAllocProducer,' acquireTransform.summary.txt || die 'summary missing transformAsync module' $?
grep -Eq '^ +event$'        acquireTransform.summary.txt || die 'summary missing event transition' $?
grep -Eq '^ +acquire (.*)$' acquireTransform.summary.txt || die 'summary missing acquire activity' $?
grep -Eq '^ +process (.*)$' acquireTransform.summary.txt || die 'summary missing process activity' $?
# ED module (Transform/TransformAsync/ExternalWork) callID must be split, not averaged together
grep -Eq '^ +process callID 1 (.*)$' acquireTransform.summary.txt || die 'summary missing ED module process callID 1' $?
grep -Eq '^ +process callID 2 (.*)$' acquireTransform.summary.txt || die 'summary missing ED module process callID 2' $?
grep -Eq '^ +acquire callID 2 (.*)$' acquireTransform.summary.txt || die 'summary missing ED module acquire callID 2' $?
# transformAllocProducer has 3 callID-only process lines (1, 2, 3); the
# 'process' activities for the event transition must be listed with
# ascending callID even though the underlying trace order is 1, 3, 2.
awk '/^Module label transformAllocProducer/{p=1} p&&/^[a-zA-Z]/&&!/^Module label transformAllocProducer/{p=0} p&&/^ +process callID [0-9]+ /{print $3}' acquireTransform.summary.txt > transformAllocProducer_callIDs.txt
printf '1\n2\n3\n' > transformAllocProducer_callIDs_expected.txt
diff transformAllocProducer_callIDs.txt transformAllocProducer_callIDs_expected.txt || die 'transformAllocProducer process callIDs not sorted ascending' $?
