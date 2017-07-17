#!/bin/bash
set -o nounset

#WorkDir='/home/dqmprolocal/filecopy' #comment out because it is defined in alivecheck_filesave.sh

[ $# -gt 1 ] || { echo Usage: $(basename $0) inputFile outputFile; exit 1; }
inputFile=$1
outputFile=$2
#echo $inputFile | grep '^Playback' > /dev/null
echo $inputFile | grep 'Playback' > /dev/null
[ $? -eq 0 ] || { echo Incorrect input file: $inputFile; exit 2; }

#outputFile=$(echo $inputFile | sed s/^Playback/DQM/)
#outputFile=$(echo $inputFile | sed s/Playback/DQM/)
[ "$inputFile" != "$outputFile" ] || { echo Input and output files are the same!; exit 3; }

# If needed set ROOT environment
#export ROOTSYS=/nfshome0/cmssw2/slc4_ia32_gcc345/lcg/root/5.18.00a-cms11/
#export ROOT_DIR=${ROOTSYS}
#export LD_LIBRARY_PATH=${ROOTSYS}/lib
#export PATH=${ROOTSYS}/bin:${PATH}

#cd /nfshome0/dqmpro/CMSSW_2_1_4/src
#eval $(scramv1 runtime -sh)
#cd -

root -l -b -q $WorkDir/sistrip_reduce_file.C++"(\"$inputFile\", \"$outputFile\")"
exit 0
