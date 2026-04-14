#!/bin/sh -ex

function die { echo $1: status $2 ;  exit $2; }

LOCAL_TEST_DIR=${SCRAM_TEST_PATH}

## Check that the default configuration fails
LD_PRELOAD="libPerfToolsAllocMonitorPreload.so" cmsRun ${LOCAL_TEST_DIR}/moduleAllocProfiler_cfg.py  > moduleAllocProfiler_none.log 2>&1  && die 'Running moduleAllocProfiler_cfg.py did not fail' 1
grep -q "moduleNames must be non-empty: ModuleAllocProfiler is intended to profile individual modules, and profiling all modules at once would be too costly." moduleAllocProfiler_none.log || die 'Failure verifying that moduleAllocProfiler_cfg.py fails with empty moduleNames' $?

## profile thingProducer (ED module)
LD_PRELOAD="libPerfToolsAllocMonitorPreload.so" cmsRun ${LOCAL_TEST_DIR}/moduleAllocProfiler_cfg.py  --edmodule > moduleAllocProfiler_edmodule.log 2>&1  || die 'Failure running moduleAllocProfiler_cfg.py --edmodule' $?

LD_PRELOAD="libPerfToolsAllocMonitorPreload.so" cmsRun ${LOCAL_TEST_DIR}/moduleAllocProfiler_cfg.py --edmodule --output "moduleAllocProfiler_edmodule_%M_%S_%I_%T.log"  > moduleAllocProfiler_edmodule_file.log 2>&1 || die 'Failure running moduleAllocProfiler_cfg.py --edmodule with file output' $?

## profile WhatsItESProducer (ES module)
LD_PRELOAD="libPerfToolsAllocMonitorPreload.so" cmsRun ${LOCAL_TEST_DIR}/moduleAllocProfiler_cfg.py --esmodule > moduleAllocProfiler_esmodule.log 2>&1 || die 'Failure running moduleAllocProfiler_cfg.py --esmodule' $?

## skip events: only events 3+ are profiled
LD_PRELOAD="libPerfToolsAllocMonitorPreload.so" cmsRun ${LOCAL_TEST_DIR}/moduleAllocProfiler_cfg.py --skipEvents  --edmodule > moduleAllocProfiler_skipEvents.log 2>&1 || die 'Failure running moduleAllocProfiler_cfg.py --skipEvents --edmodule' $?
