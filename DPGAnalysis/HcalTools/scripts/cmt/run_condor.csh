#!/bin/tcsh
echo "Starting job on " `date` #Date/time of start of job
echo "Running on: `uname -a`" #Condor job is running on this node
echo "System software: `cat /etc/redhat-release`" #Operating System on that node
source /cvmfs/cms.cern.ch/cmsset_default.csh  ## if a bash script, use .sh instead of .csh

echo "myStart run_condor.csh"

echo "myStart: set dir pwd"
set m=`pwd`

cd ${m}
#setenv SCRAM_ARCH slc7_amd64_gcc10
#cmsrel CMSSW_12_3_4
#cd CMSSW_12_3_4/src 
#setenv SCRAM_ARCH ec8_amd64_gcc10
setenv SCRAM_ARCH ec8_amd64_gcc11
cmsrel CMSSW_13_0_0_pre1
cd CMSSW_13_0_0_pre1/src 

mkdir DPGAnalysis
cd DPGAnalysis

mkdir HcalTools
cd HcalTools

mv ../../../../BuildFile.xml .
ls

mkdir interface
mv ../../../../CMTRawAnalyzer.h interface/.
mkdir src
mv ../../../../CMTRawAnalyzer.cc src/.

cd src
ls
eval `scramv1 runtime -csh` # cmsenv is an alias not on the workers
echo "myStart: scram b"
scram b 
echo "myStart: scram b DONE"

cd -
mkdir test
cd test
cp ../../../../../* .
ls

setenv X509_USER_PROXY ${4}
voms-proxy-info -all
voms-proxy-info -all -file ${4}

./mkcfg_new120.csh ${1}
ls PYTHON_${1}/*py

################################################################ loop:  
echo "myStart: loop in run_condor.csh"
if( ${status} == "0" ) then
foreach i (`ls PYTHON_${1}`)

set j=`echo ${i} | awk -F _ '{print $2}'`
set k=`echo ${i} | awk -F _ '{print $3}'`

echo ${m} ${1} ${i} ${j} ${k} 

eval `scramv1 runtime -csh` # cmsenv is an alias not on the workers 
cmsRun PYTHON_${1}/Reco_${j}_${k}_cfg.py 

### Copy output files to EOS
### xrdcp -f Global_${j}_${k}.root /eos/cms/store/user/zhokin/CMT/test/Global_${j}_${k}.root
#eoscp Global_${j}_${k}.root /eos/cms/store/user/zhokin/CMT/RootFilesToBeMarched/2022/Global_${j}_${k}.root
#eoscp Global_${j}_${k}.root /eos/cms/store/user/zhokin/CMT/RootFilesToBeMarched/2022/run3a1/Global_${j}_${k}.root
eoscp Global_${j}_${k}.root /eos/cms/store/user/zhokin/CMT/RootFilesToBeMarched/2023/Global_${j}_${k}.root
#eoscp Global_${j}_${k}.root /eos/cms/store/user/zhokin/CMT/test/Global_${j}_${k}.root

################################################################
end
else
echo "Problem: No jobs are created: check PYTHON_${1} directory: Notify developpers"
endif

