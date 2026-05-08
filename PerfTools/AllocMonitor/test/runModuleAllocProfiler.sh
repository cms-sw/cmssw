#!/bin/sh -ex

function die { echo $1: status $2 ;  exit $2; }

function require_profiler_files {
  local PREFIX=$1
  local EMPTY_SUFFIXES=$2
  local DONTCARE_SUFFIXES=$3
  if [ "$EMPTY_SUFFIXES" = "all-empty" ]; then
    EMPTY_SUFFIXES="added alloc atMaxActual churn churnalloc dealloc"
  fi
  for SUFFIX in added alloc atMaxActual churn churnalloc dealloc; do
    [ -f "${PREFIX}_${SUFFIX}.log" ] || die "Missing file ${PREFIX}_${SUFFIX}.log" 1
    case " $DONTCARE_SUFFIXES " in
      *" $SUFFIX "*) ;; # file exists; content not checked
      *)
        case " $EMPTY_SUFFIXES " in
          *" $SUFFIX "*)
            if grep -q -v '^#' "${PREFIX}_${SUFFIX}.log"; then
              die "Expected empty log ${PREFIX}_${SUFFIX}.log" 1
            fi ;;
          *)
            grep -q -v '^#' "${PREFIX}_${SUFFIX}.log" || die "Expected non-empty log ${PREFIX}_${SUFFIX}.log" 1
            if [ "$SUFFIX" = "dealloc" ]; then
              QUANTS="count actual"
            else
              QUANTS="count requested actual"
            fi
            for QUANT in $QUANTS; do
              edmAllocProfilerFoldStacks.py -q $QUANT "${PREFIX}_${SUFFIX}.log" > "${PREFIX}_${SUFFIX}_${QUANT}.log" 2>&1 || die "Failure folding stacks for ${PREFIX}_${SUFFIX}.log" $?
            done ;;
        esac ;;
    esac
  done
}

function require_no_profiler_files {
  local PREFIX=$1
  for SUFFIX in added alloc atMaxActual churn churnalloc dealloc; do
    if [ -f "${PREFIX}_${SUFFIX}.log" ]; then
      die "Unexpected file ${PREFIX}_${SUFFIX}.log" 1
    fi
  done
}

LOCAL_TEST_DIR=${SCRAM_TEST_PATH}

function test_none {
  ## Check that the default configuration fails
  LD_PRELOAD="libPerfToolsAllocMonitorPreload.so" cmsRun ${LOCAL_TEST_DIR}/moduleAllocProfiler_cfg.py  > moduleAllocProfiler_none.log 2>&1  && die 'Running moduleAllocProfiler_cfg.py did not fail' 1
  grep -q "moduleNames must be non-empty: ModuleAllocProfiler is intended to profile individual modules, and profiling all modules at once would be too costly." moduleAllocProfiler_none.log || die 'Failure verifying that moduleAllocProfiler_cfg.py fails with empty moduleNames' $?
}

function test_edmodule {
  ## profile select EDModules
  LD_PRELOAD="libPerfToolsAllocMonitorPreload.so" cmsRun ${LOCAL_TEST_DIR}/moduleAllocProfiler_cfg.py  --edmodule > moduleAllocProfiler_edmodule.log 2>&1  || die 'Failure running moduleAllocProfiler_cfg.py --edmodule' $?
  [ $(grep -c "Ending tracing" moduleAllocProfiler_edmodule.log) -eq 23 ] || die 'Expected 12 instances of "Ending tracing" in moduleAllocProfiler_edmodule.log' 1

  PREFIX="moduleAllocProfiler_edmodule_"
  LD_PRELOAD="libPerfToolsAllocMonitorPreload.so" cmsRun ${LOCAL_TEST_DIR}/moduleAllocProfiler_cfg.py --edmodule --output "${PREFIX}%M_%S_%C_%I_%T.log"  > moduleAllocProfiler_edmodule_file.log 2>&1 || die 'Failure running moduleAllocProfiler_cfg.py --edmodule with file output' $?
  COMMON="${PREFIX}thingProducer_module"
  require_profiler_files "${COMMON}Construction_0_0" ""
  require_profiler_files "${COMMON}BeginJob_0_0" all-empty
  require_profiler_files "${COMMON}BeginStream_0_0" all-empty
  require_profiler_files "${COMMON}GlobalBeginRun_0_0" ""
  require_profiler_files "${COMMON}GlobalBeginLumi_0_0" ""
  require_profiler_files "${COMMON}Event_0_0" ""
  require_profiler_files "${COMMON}Event_0_1" ""
  require_profiler_files "${COMMON}Event_0_2" ""
  require_profiler_files "${COMMON}GlobalEndLumi_0_0" ""
  require_profiler_files "${COMMON}GlobalEndRun_0_0" ""
  require_profiler_files "${COMMON}EndStream_0_0" all-empty
  require_profiler_files "${COMMON}EndJob_0_0" all-empty
  COMMON="${PREFIX}externalWorkProducer_module"
  require_profiler_files "${COMMON}Construction_0_0" ""
  require_profiler_files "${COMMON}BeginJob_0_0" all-empty
  require_profiler_files "${COMMON}BeginStream_0_0" all-empty
  require_profiler_files "${COMMON}EventAcquire_0_0" ""
  require_profiler_files "${COMMON}Event_0_0" ""
  require_profiler_files "${COMMON}EventAcquire_0_1" ""
  require_profiler_files "${COMMON}Event_0_1" ""
  require_profiler_files "${COMMON}EventAcquire_0_2" ""
  require_profiler_files "${COMMON}Event_0_2" ""
  require_profiler_files "${COMMON}EndStream_0_0" all-empty
  require_profiler_files "${COMMON}EndJob_0_0" all-empty
}

function test_esmodule {
  ## profile select ESModules
  LD_PRELOAD="libPerfToolsAllocMonitorPreload.so" cmsRun ${LOCAL_TEST_DIR}/moduleAllocProfiler_cfg.py --esmodule > moduleAllocProfiler_esmodule.log 2>&1 || die 'Failure running moduleAllocProfiler_cfg.py --esmodule' $?
  [ $(grep -c "Ending tracing" moduleAllocProfiler_esmodule.log) -eq 8 ] || die 'Expected 8 instances of "Ending tracing" in moduleAllocProfiler_esmodule.log' 1

  PREFIX="moduleAllocProfiler_esmodule_"
  LD_PRELOAD="libPerfToolsAllocMonitorPreload.so" cmsRun ${LOCAL_TEST_DIR}/moduleAllocProfiler_cfg.py --esmodule --output "${PREFIX}%M_%S_%C_%I_%T.log" > moduleAllocProfiler_esmodule_file.log 2>&1 || die 'Failure running moduleAllocProfiler_cfg.py --esmodule with file output' $?
  COMMON="${PREFIX}WhatsItESProducer_esModule"
  require_profiler_files "${COMMON}Construction_0_0" ""
  require_profiler_files "${COMMON}_0_0" "churn churnalloc dealloc"
  require_profiler_files "${COMMON}_1_0" "churn churnalloc dealloc"
  require_profiler_files "${COMMON}_2_0" "churn churnalloc dealloc"
  require_profiler_files "${COMMON}_3_0" "churn churnalloc dealloc"
  require_profiler_files "${COMMON}_4_0" all-empty

  COMMON="${PREFIX}acquireIntESProducer_esModule"
  require_profiler_files "${COMMON}Construction_0_0" ""
  require_profiler_files "${COMMON}_1_0" "churn churnalloc dealloc"
}

function test_source {
  ## Profile source
  LD_PRELOAD="libPerfToolsAllocMonitorPreload.so" cmsRun ${LOCAL_TEST_DIR}/moduleAllocProfiler_cfg.py --source > moduleAllocProfiler_source.log 2>&1 || die 'Failure running moduleAllocProfiler_cfg.py --source' $?
  [ $(grep -c "Ending tracing" moduleAllocProfiler_source.log) -eq 13 ] || die 'Expected 13 instances of "Ending tracing" in moduleAllocProfiler_source.log' 1

  PREFIX="moduleAllocProfiler_source_"
  LD_PRELOAD="libPerfToolsAllocMonitorPreload.so" cmsRun ${LOCAL_TEST_DIR}/moduleAllocProfiler_cfg.py --source --output "${PREFIX}%M_%S_%C_%I_%T.log" > moduleAllocProfiler_source_file.log 2>&1 || die 'Failure running moduleAllocProfiler_cfg.py --source with file output' $?
  COMMON="${PREFIX}source_source"
  require_profiler_files "${COMMON}Construction_0_0" ""
  require_profiler_files "${COMMON}NextTransition_0_0" all-empty
  require_profiler_files "${COMMON}NextTransition_0_1" "churn churnalloc dealloc"
  require_profiler_files "${COMMON}NextTransition_0_2" "churn churnalloc dealloc"
  require_profiler_files "${COMMON}NextTransition_0_3" all-empty
  require_profiler_files "${COMMON}NextTransition_0_4" all-empty
  require_profiler_files "${COMMON}NextTransition_0_5" all-empty
  require_profiler_files "${COMMON}NextTransition_0_6" "added alloc atMaxActual churn churnalloc"
  require_profiler_files "${COMMON}Run_0_0" ""
  require_profiler_files "${COMMON}Lumi_0_0" "churn churnalloc"
  require_profiler_files "${COMMON}Event_0_0" ""
  require_profiler_files "${COMMON}Event_0_1" "" "added"
  require_profiler_files "${COMMON}Event_0_2" "" "added"
}

function test_out {
  ## Profile OutputModule
  LD_PRELOAD="libPerfToolsAllocMonitorPreload.so" cmsRun ${LOCAL_TEST_DIR}/moduleAllocProfiler_cfg.py --out > moduleAllocProfiler_out.log 2>&1 || die 'Failure running moduleAllocProfiler_cfg.py --out' $?
  [ $(grep -c "Ending tracing" moduleAllocProfiler_out.log) -eq 15 ] || die 'Expected 15 instances of "Ending tracing" in moduleAllocProfiler_out.log' 1

  PREFIX="moduleAllocProfiler_out_"
  LD_PRELOAD="libPerfToolsAllocMonitorPreload.so" cmsRun ${LOCAL_TEST_DIR}/moduleAllocProfiler_cfg.py --out --output "${PREFIX}%M_%S_%C_%I_%T.log" > moduleAllocProfiler_out_file.log 2>&1 || die 'Failure running moduleAllocProfiler_cfg.py --out with file output' $?
  COMMON="${PREFIX}out_module"
  require_profiler_files "${COMMON}Construction_0_0" ""
  require_profiler_files "${COMMON}BeginJob_0_0" all-empty
  require_profiler_files "${COMMON}BeginStream_0_0" all-empty
  require_profiler_files "${COMMON}GlobalBeginRun_0_0" all-empty
  require_profiler_files "${COMMON}GlobalBeginLumi_0_0" "added"
  require_profiler_files "${COMMON}Event_0_0" ""
  require_profiler_files "${COMMON}Event_0_1" "" "added" # some times added is empty and sometimes not
  require_profiler_files "${COMMON}Event_0_2" "" "added" # some times added is empty and sometimes not
  require_profiler_files "${COMMON}WriteLumi_0_0" "added"
  require_profiler_files "${COMMON}WriteRun_0_0" all-empty
  require_profiler_files "${COMMON}WriteProcessBlock_0_0" all-empty
  require_profiler_files "${COMMON}EndStream_0_0" all-empty
  require_profiler_files "${COMMON}EndJob_0_0" all-empty
}

function test_skipEvents {
  ## skip events: only events 3+ are profiled
  LD_PRELOAD="libPerfToolsAllocMonitorPreload.so" cmsRun ${LOCAL_TEST_DIR}/moduleAllocProfiler_cfg.py --skipEvents  --edmodule > moduleAllocProfiler_skipEvents.log 2>&1 || die 'Failure running moduleAllocProfiler_cfg.py --skipEvents --edmodule' $?
  [ $(grep -c "Ending tracing" moduleAllocProfiler_skipEvents.log) -eq 9 ] || die 'Expected 9 instances of "Ending tracing" in moduleAllocProfiler_skipEvents.log' 1

  PREFIX="moduleAllocProfiler_skipEvents_"
  LD_PRELOAD="libPerfToolsAllocMonitorPreload.so" cmsRun ${LOCAL_TEST_DIR}/moduleAllocProfiler_cfg.py --skipEvents  --edmodule --output "${PREFIX}%M_%S_%C_%I_%T.log" > moduleAllocProfiler_skipEvents_file.log 2>&1 || die 'Failure running moduleAllocProfiler_cfg.py --skipEvents --edmodule with file output' $?
  COMMON="${PREFIX}thingProducer_module"
  require_no_profiler_files "${COMMON}Construction_0_0"
  require_no_profiler_files "${COMMON}BeginJob_0_0"
  require_no_profiler_files "${COMMON}BeginStream_0_0"
  require_no_profiler_files "${COMMON}GlobalBeginRun_0_0"
  require_no_profiler_files "${COMMON}GlobalBeginLumi_0_0"
  require_no_profiler_files "${COMMON}Event_0_0"
  require_no_profiler_files "${COMMON}Event_0_1"
  require_profiler_files "${COMMON}Event_0_2" ""
  require_profiler_files "${COMMON}GlobalEndLumi_0_0" ""
  require_profiler_files "${COMMON}GlobalEndRun_0_0" ""
  require_profiler_files "${COMMON}EndStream_0_0" all-empty
  require_profiler_files "${COMMON}EndJob_0_0" all-empty
  COMMON="${PREFIX}externalWorkProducer_module"
  require_no_profiler_files "${COMMON}Construction_0_0"
  require_no_profiler_files "${COMMON}BeginJob_0_0"
  require_no_profiler_files "${COMMON}BeginStream_0_0"
  require_no_profiler_files "${COMMON}EventAcquire_0_0"
  require_no_profiler_files "${COMMON}Event_0_0"
  require_no_profiler_files "${COMMON}EventAcquire_0_1"
  require_no_profiler_files "${COMMON}Event_0_1"
  require_profiler_files "${COMMON}EventAcquire_0_2" ""
  require_profiler_files "${COMMON}Event_0_2" ""
  require_profiler_files "${COMMON}EndStream_0_0" all-empty
  require_profiler_files "${COMMON}EndJob_0_0" all-empty
}

function test_clearEvent {
  LD_PRELOAD="libPerfToolsAllocMonitorPreload.so" cmsRun ${LOCAL_TEST_DIR}/moduleAllocProfiler_cfg.py --clearEvent  > moduleAllocProfiler_clearEvent.log 2>&1 || die 'Failure running moduleAllocProfiler_cfg.py --clearEvent' $?
  [ $(grep -c "Ending tracing" moduleAllocProfiler_clearEvent.log) -eq 3 ] || die 'Expected 3 instances of "Ending tracing" in moduleAllocProfiler_clearEvent.log' 1

  PREFIX="moduleAllocProfiler_clearEvent_"
  LD_PRELOAD="libPerfToolsAllocMonitorPreload.so" cmsRun ${LOCAL_TEST_DIR}/moduleAllocProfiler_cfg.py --clearEvent  --output "${PREFIX}%M_%S_%C_%I_%T.log" > moduleAllocProfiler_clearEvent_file.log 2>&1 || die 'Failure running moduleAllocProfiler_cfg.py --clearEvent with file output' $?
  COMMON="${PREFIX}ClearEvent_clearEvent"
  require_profiler_files "${COMMON}_0_0" "added"
  require_profiler_files "${COMMON}_0_1" "added"
  require_profiler_files "${COMMON}_0_2" "added"
}

case "$1" in
  none)       test_none ;;
  edmodule)   test_edmodule ;;
  esmodule)   test_esmodule ;;
  source)     test_source ;;
  out)        test_out ;;
  skipEvents) test_skipEvents ;;
  clearEvent) test_clearEvent ;;
  *) die "Unknown test '$1': valid options are none, edmodule, esmodule, source, out, skipEvents, clearEvent" 1 ;;
esac
