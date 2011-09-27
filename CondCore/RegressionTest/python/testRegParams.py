import sys
import os
import shlex, subprocess

cmdList = """
SCRAM_ARCH=slc5_amd64_gcc451
CYC=4_2
RELEASE=$(scram l CMSSW | grep CMSSW_$CYC | tail -2 | head -1 | awk '{print $2}')
RPATH=$(scram l CMSSW | grep CMSSW_$CYC | tail -1 | awk '{print $2}')
REF_RELEASE=$(scram l CMSSW | grep CMSSW_$CYC | tail -4 | head -1 | awk '{print $2}')
REF_PATH=$(scram l CMSSW | grep CMSSW_$CYC | tail -4 | head -2 | tail -1 | awk '{print $2}')
echo "python testRegression.py -R $RELEASE -A $SCRAM_ARCH -P $RPATH --R $REF_RELEASE --A $SCRAM_ARCH --P $REF_PATH"  
"""
if cmdList != "":
	pipe = subprocess.Popen(cmdList, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
	stdout_value = pipe.communicate()[0]
	print '\Output:', stdout_value