
export barycenter_bit=$1
export width_bit=$2
export avgCharge_bit=$3
export evn=$4
export thread=$5
export remove=$6
export output=/scratch/$(whoami)/$7
export tmp=/home/users/$(whoami)/tmp/$7
export cluster=$8
export git_branch=$9
export strip_charge_cut=${10}

echo $tmp
echo $output
if  ! [ -d $output ]; then
  eval "mkdir -p $output"
fi

if [ -d ${tmp} ]; then
  echo "removing"
  eval "rm -rf ${tmp}"
fi

eval "mkdir -p ${tmp}"
eval "cd ${tmp}"

eval ". /cvmfs/cms.cern.ch/cmsset_default.sh"
eval "cmsrel CMSSW_14_1_5"
eval "cd CMSSW_14_1_5/src/"
eval "cmsenv"
eval "git cms-init"
eval "git checkout -b ${git_branch}"
eval "git pull git@github.com:saswatinandan/cmssw.git ${git_branch}"
eval "git cms-addpkg Configuration/Eras DataFormats/SiStripCluster"
eval "git cms-addpkg RecoLocalTracker/SiStripClusterizer" 
eval "scram b -j 8"
eval "cd RecoLocalTracker/SiStripClusterizer/test"

eval "git remote -v >  git.log"
eval "git branch -v >>  git.log"

export current_dir=$(pwd)
echo 'curent dir: ', $current_dir

echo $strip_charge_cut
echo "python3 run_cmsDriver.py -b $barycenter_bit -w $width_bit -a $avgCharge_bit -n $evn -t $thread -c $cluster -s $strip_charge_cut"
python3 run_cmsDriver.py -b $barycenter_bit -w $width_bit -a $avgCharge_bit -n $evn -t $thread -c $cluster -s $strip_charge_cut

eval "cd ${output}"
eval "cp $current_dir/*png ."
eval "cp $current_dir/*step*root ."
eval "cp $current_dir/*study*root ."
eval "cp $current_dir/outputPhy*root ."
eval "cp $current_dir/*log ."

echo "remove", $remove
if [ $remove -eq 1 ]; then
  echo $tmp
  eval "rm -rf ${tmp}"
fi
