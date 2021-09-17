#!/bin/bash

LOCALTOP=$1
DEVICE=$2

# the test is not possible if:
# 1. GPU not available (only if GPU test requested) / avx instructions not supported (needed for singularity on CPU)
# 2. singularity not found or not usable
# 3. inside singularity container w/o unprivileged user namespace enabled (needed for singularity-in-singularity)
# so just return true in those cases

if [ "$DEVICE" = "GPU" ]; then
	if nvidia-smi -L; then
		echo "has GPU"
	else
		echo "missing GPU"
		exit 0
	fi
else
	if grep -q avx /proc/cpuinfo; then
		echo "has avx"
	else
		echo "missing avx"
		exit 0
	fi
fi

if type singularity >& /dev/null; then
	echo "has singularity"
else
	echo "missing singularity"
	exit 0
fi

if [ -n "$SINGULARITY_CONTAINER" ]; then
	if grep -q "^allow setuid = no" /etc/singularity/singularity.conf && unshare -U echo >/dev/null 2>&1; then
		echo "has unprivileged user namespace support"
	else
		echo "missing unprivileged user namespace support"
		exit 0
	fi
fi

tmpFile=$(mktemp -p ${LOCALTOP} SonicTritonTestXXXXXXXX.log)
cmsRun ${LOCALTOP}/src/HeterogeneousCore/SonicTriton/test/tritonTest_cfg.py modules=TritonGraphProducer,TritonGraphFilter,TritonGraphAnalyzer maxEvents=2 unittest=1 verbose=1 device=${DEVICE} testother=1 >& $tmpFile
CMSEXIT=$?

cat $tmpFile

STOP_COUNTER=0
while ! LOGFILE="$(ls -rt ${LOCALTOP}/log_triton_server_instance*.log 2>/dev/null | tail -n 1)" && [ "$STOP_COUNTER" -lt 5 ]; do
	STOP_COUNTER=$((STOP_COUNTER+1))
	sleep 5
done

if [ -n "$LOGFILE" ]; then
	echo -e '\n=====\nContents of '$LOGFILE':\n=====\n'
	cat "$LOGFILE"
fi

if grep -q "Socket closed" $tmpFile; then
	echo "Transient server error (not caused by client code)"
	CMSEXIT=0
fi

rm $tmpFile
exit $CMSEXIT
