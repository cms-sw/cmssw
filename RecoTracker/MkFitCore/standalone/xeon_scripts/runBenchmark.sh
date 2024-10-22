#! /bin/bash

##### Command Line Input #####
suite=${1:-"forPR"} # which set of benchmarks to run: full, forPR, forConf
useARCH=${2:-0}  # 0 phi3 only, 1 lnx only, 2 phi3 + lnx, 3 phi123, 4 phi123 + lnx
lnxuser=${3:-${USER}}

##### Initialize Benchmarks #####
source xeon_scripts/common-variables.sh ${suite} ${useARCH} ${lnxuser}
source xeon_scripts/init-env.sh
make distclean

##### Check Settings #####
assert_settings=true
echo "--------Showing System Settings--------"
echo "turbo status: "$(cat /sys/devices/system/cpu/intel_pstate/no_turbo)
echo "scaling governor setting: "$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)
echo "--------End System Settings ------------"
if ${assert_settings}
then
echo "Ensuring correct settings"
if [[ $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor) != "performance" ]]
then
echo "performance mode is OFF. Exiting"
exit 1
fi
if [[ $(cat /sys/devices/system/cpu/intel_pstate/no_turbo) == "0" ]]
then
echo "Turbo is ON. Exiting"
exit 1
fi
fi
sleep 3 ## so you can see the settings

##### Launch Tests #####
if [[ ${useARCH} -eq 1 ]] || [[ ${useARCH} -eq 2 ]] || [[ ${useARCH} -eq 4 ]]
then
echo "Tar and send to LNX7188"
./xeon_scripts/tarAndSendToRemote.sh LNX-G ${suite} ${useARCH} ${lnxuser}
if [ $? -eq 1 ]; then
echo "lnx7188 has bad settings. Please fix them and try again"
exit 1
fi

echo "Run benchmarking on LNX7188 concurrently with SKL-SP benchmarks" 
./xeon_scripts/benchmark-cmssw-ttbar-fulldet-build-remote.sh LNX-G ${suite} ${useARCH} ${lnxuser} >& benchmark_lnx-g_dump.txt &

echo "Tar and send to LNX4108"
./xeon_scripts/tarAndSendToRemote.sh LNX-S ${suite} ${useARCH} ${lnxuser}
if [ $? -eq 1 ]; then
echo "lnx4108 has bad settings. Please fix them and try again"
exit 1
fi

echo "Run benchmarking on LNX4108 concurrently with SKL-SP benchmarks" 
./xeon_scripts/benchmark-cmssw-ttbar-fulldet-build-remote.sh LNX-S ${suite} ${useARCH} ${lnxuser} >& benchmark_lnx-s_dump.txt &
fi

if [[ ${useARCH} -eq 3 ]] || [[ ${useARCH} -eq 4 ]]
then

echo "Tar and send to KNL"
./xeon_scripts/tarAndSendToRemote.sh KNL ${suite} ${useARCH} ${lnxuser}
if [ $? -eq 1 ]; then
echo "KNL has bad settings. Please fix them and try again"
exit 1
fi

echo "Run benchmarking on KNL concurrently with SKL-SP benchmarks" 
./xeon_scripts/benchmark-cmssw-ttbar-fulldet-build-remote.sh KNL ${suite} ${useARCH} ${lnxuser} >& benchmark_knl_dump.txt &

echo "Tar and send to SNB"
./xeon_scripts/tarAndSendToRemote.sh SNB ${suite} ${useARCH} ${lnxuser}
if [ $? -eq 1 ]; then
echo "SNB has bad settings. Please fix them and try again"
exit 1
fi

echo "Run benchmarking on SNB concurrently with SKL-SP benchmarks" 
./xeon_scripts/benchmark-cmssw-ttbar-fulldet-build-remote.sh SNB ${suite} ${useARCH} ${lnxuser} >& benchmark_snb_dump.txt &
fi

if [[ ${useARCH} -eq 0 ]] || [[ ${useARCH} -eq 2 ]] || [[ ${useARCH} -eq 3 ]] || [[ ${useARCH} -eq 4 ]]
then
echo "Run benchmarking on SKL-SP"
./xeon_scripts/benchmark-cmssw-ttbar-fulldet-build.sh SKL-SP ${suite} ${useARCH} ${lnxuser} 
fi

##### Validation tests #####
echo "Running ROOT based validation"
./val_scripts/validation-cmssw-benchmarks.sh ${suite} --mtv-like-val

if [[ ${useARCH} -eq 1 ]] || [[ ${useARCH} -eq 2 ]]
then
echo "Waiting for LNX-G and LNX-S"
elif [[ ${useARCH} -eq 3 ]] 
then
echo "Waiting for KNL and SNB"
elif  [[ ${useARCH} -eq 4 ]]
then 
echo "Waiting for LNX-G, LNX-S, KNL, and SNB"
fi
wait

##### Benchmark Plots #####
echo "Producing benchmarking plots"
./plotting/benchmarkPlots.sh ${suite} ${useARCH} ${lnxuser} 

##### Plots from Text Files #####
echo "Producing plots from text files"
./plotting/textDumpPlots.sh ${suite} ${useARCH} ${lnxuser} 

##### Final cleanup #####
make distclean

##### Final message #####
echo "Finished benchmarking and validation suite!"
