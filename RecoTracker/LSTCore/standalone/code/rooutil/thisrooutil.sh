#export SCRAM_ARCH=slc6_amd64_gcc530   # or whatever scram_arch you need for your desired CMSSW release
#export CMSSW_VERSION=CMSSW_9_2_0
#source /cvmfs/cms.cern.ch/cmsset_default.sh
#cd /cvmfs/cms.cern.ch/$SCRAM_ARCH/cms/cmssw/$CMSSW_VERSION/src
#eval `scramv1 runtime -sh`
#cd - > /dev/null
#
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export LD_LIBRARY_PATH=$DIR:$LD_LIBRARY_PATH
export PYTHONPATH="$DIR:$PYTHONPATH"
export PATH=$DIR:$PATH
