#! /bin/bash
TOOL=$CMSSW_BASE/config/toolbox/$SCRAM_ARCH/tools/selected/cuda.xml

# enumerate the supported streaming multiprocessor (sm) compute capabilites
DOTS=$(cudaComputeCapabilities | awk '{ print $2 }' | sort -u)
CAPS=$(echo $DOTS | sed -e's#\.*##g')

# remove existing capabilities
sed -i $TOOL -e'\#<flags CUDA_FLAGS="-gencode arch=compute_..,code=sm_.."/>#d'

# add support for the capabilities found on this machine
for CAP in $CAPS; do
  sed -i $TOOL -e"\#</client>#a\  <flags CUDA_FLAGS=\"-gencode arch=compute_$CAP,code=sm_$CAP\"/>"
done

# reconfigure the cuda.xml tool
scram setup cuda

echo "SCRAM configured to support CUDA streaming multiprocessor architectures $DOTS"
