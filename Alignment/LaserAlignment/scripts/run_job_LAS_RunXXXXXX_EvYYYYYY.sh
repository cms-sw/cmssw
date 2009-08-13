#!/bin/sh
echo "Checking the environment settings ..."
#check if the cmssw environment is set;
#the `` saves in a variable the output of the command
cmsRun_file=`which cmsRun`
#get the path
#file_home=
#set configuration file
file_name=t0ProducerStandalone_RunXXXXXX_EvYYYYYY_F0
file_out=TkAlLAS_RunXXXXXX_EvYYYYYY_F0
file_ext1=.py
file_ext2=.log
file_ext3=.root
file_ext4=.dqm.root

if [ -z $cmsRun_file -o $cmsRun_file = "cmsRun not found" ]
then
    echo "E: The cmssw environment is NOT set!"
    echo "H: Please make followings:"
    echo "% cd $HOME"
    echo "% source setup_cmssw.sh"
    echo "H: and try again:"
    echo "% ./run_on_LSF.sh"
else
    cp -r $HOME/cms/CMSSW_3_2_3LAS .
    cd CMSSW_3_2_3LAS/src
    eval `scramv1 runtime -sh` 
    scramv1 b ProjectRename
    ./make_cmssw.sh
    echo "Environment set, the job will be started ..."
    cd Alignment/LaserAlignment/scripts
    cmsRun py_config/$file_name$file_ext1 >& $file_name$file_ext2 
    echo "Job has been finished!"
    scp $file_name$file_ext2 lxplus218:/tmp/aperiean/RunXXXXXX/RunXXXXXX_EvYYYYYY
    scp $file_out$file_ext3  lxplus218:/tmp/aperiean/RunXXXXXX/RunXXXXXX_EvYYYYYY
    scp $file_out$file_ext4  lxplus218:/tmp/aperiean/RunXXXXXX/RunXXXXXX_EvYYYYYY
fi
