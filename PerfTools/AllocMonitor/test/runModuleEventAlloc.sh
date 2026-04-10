#!/bin/sh -ex

function die { echo $1: status $2 ;  exit $2; }

LOCAL_TEST_DIR=${SCRAM_TEST_PATH}

LD_PRELOAD="libPerfToolsAllocMonitorPreload.so" cmsRun ${LOCAL_TEST_DIR}/moduleEventAlloc_cfg.py || die 'Failure using moduleEventAlloc_cfg.py' $?

ORIGLOGLINES=$(grep -v "#" moduleEventAlloc.log | grep -v "@" | wc -l)

THINGANALYZERID=$(grep edmtest::ThingAnalyzer moduleEventAlloc.log | awk '{print $4}')
grep "M ${THINGANALYZERID} " moduleEventAlloc.log | cut -d ' ' -f6- > thingAnalyzerUnmatched.txt 2>&1
diff thingAnalyzerUnmatched.txt ${LOCAL_TEST_DIR}/unittest_output/moduleEventAllocMonitor_thingAnalyzerUnmatched.txt || die "differences in thingAnalyzerUnmatched" $?

edmModuleEventAllocMonitorAnalyze.py --grew moduleEventAlloc.log > grew.txt 2>&1
LINES=$(cat grew.txt | wc -l)
echo $LINES
if [ "$LINES" -ne 1 ]; then
    echo "Some modules were reported to retain memory over the job"
    cat grew.txt
    exit 1
fi

edmModuleEventAllocMonitorAnalyze.py --retained moduleEventAlloc.log > retained.txt 2>&1
LINES=$(cat retained.txt | wc -l)
if [ "$LINES" -ne 2 ]; then
    echo "Unexpected number of modules reported to retain memory from one event to another. Expected AcquireIntStreamProducer"
    cat retained.txt
    exit 1
fi
grep -q AcquireIntStreamProducer retained.txt || die 'AcquireIntStreamProducer not reported to retain memory from one event to another.' $?

for F in product tempSize nTemp; do
    edmModuleEventAllocMonitorAnalyze.py --${F} moduleEventAlloc.log > ${F}.txt 2>&1
    LINES=$(cat ${F}.txt | wc -l)
    if [ "$LINES" -ne 6 ]; then
        echo "Unexpected number of modules in ${F}. Expected 5 modules"
        cat ${F}.txt
        exit 1
    fi
    grep -q IntProducer ${F}.txt || die 'IntProducer not reported ${F}.txt' $?
    grep -q ThingProducer ${F}.txt || die 'ThingProducer not reported ${F}.txt' $?
    grep -q AcquireIntStreamProducer ${F}.txt || die 'AcquireIntStreamProducer not reported ${F}.txt' $?
    grep -q OtherThingProducer ${F}.txt || die 'OtherThingProducer not reported ${F}.txt' $?
done

############### only 1 ED module kept
PREFIX="edmodule"
LD_PRELOAD="libPerfToolsAllocMonitorPreload.so" cmsRun ${LOCAL_TEST_DIR}/moduleEventAlloc_cfg.py --edmodule --output ${PREFIX}_moduleEventAlloc.log || die 'Failure using moduleEventAlloc_cfg.py --edmodule' $?

edmModuleEventAllocMonitorAnalyze.py --grew ${PREFIX}_moduleEventAlloc.log > ${PREFIX}_grew.txt 2>&1
LINES=$(cat ${PREFIX}_grew.txt | wc -l)
echo $LINES
if [ "$LINES" -ne 1 ]; then
    echo "Some modules were reported to retain memory over the job"
    cat ${PREFIX}_grew.txt
    exit 1
fi

edmModuleEventAllocMonitorAnalyze.py --retained ${PREFIX}_moduleEventAlloc.log > ${PREFIX}_retained.txt 2>&1
LINES=$(cat ${PREFIX}_retained.txt | wc -l)
if [ "$LINES" -ne 1 ]; then
    echo "Some modules were reported to retain memory from one event to another"
    cat ${PREFIX}_retained.txt
    exit 1
fi

for F in product tempSize nTemp; do
    edmModuleEventAllocMonitorAnalyze.py --${F} ${PREFIX}_moduleEventAlloc.log > ${PREFIX}_${F}.txt 2>&1
    LINES=$(cat ${PREFIX}_${F}.txt | wc -l)
    if [ "$LINES" -ne 2 ]; then
        echo "Unexpected number of modules in ${F}. Expected 1 modules"
        cat ${PREFIX}_${F}.txt
        exit 1
    fi
    grep -q ThingProducer ${PREFIX}_${F}.txt || die 'ThingProducer not reported ${PREFIX}_${F}.txt' $?
done

############## skip events
PREFIX="skip"
LD_PRELOAD="libPerfToolsAllocMonitorPreload.so" cmsRun ${LOCAL_TEST_DIR}/moduleEventAlloc_cfg.py --skipEvents --output ${PREFIX}_moduleEventAlloc.log || die 'Failure using moduleEventAlloc_cfg.py --skipEvents' $?

SKIPLOGLINES=$(grep -v "#" ${PREFIX}_moduleEventAlloc.log | grep -v "@" | wc -l)

if [ $ORIGLOGLINES -le $SKIPLOGLINES ]; then
    echo "Skipping events resulted in ${SKIPLOGLINES} entries in the log, whereas original log resulted in ${ORIGLOGLINES}"
    exit 1
fi


############## multiple threads, mostly just that it technically runs
PREFIX="mt"
LD_PRELOAD="libPerfToolsAllocMonitorPreload.so" cmsRun ${LOCAL_TEST_DIR}/moduleEventAlloc_cfg.py --maxEvents 64 --threads 8 --output ${PREFIX}_moduleEventAlloc.log || die 'Failure using moduleEventAlloc_cfg.py --threads 8' $?


for F in grew retained product tempSize nTemp; do
    edmModuleEventAllocMonitorAnalyze.py --${F} ${PREFIX}_moduleEventAlloc.log > ${PREFIX}_${F}.txt 2>&1
    LINES=$(cat ${PREFIX}_${F}.txt | wc -l)
    if [ "$LINES" -eq 0 ]; then
        echo "${PREFIX}_${F}.txt is empty"
        cat ${PREFIX}_${F}.txt
        exit 1
    fi
done
