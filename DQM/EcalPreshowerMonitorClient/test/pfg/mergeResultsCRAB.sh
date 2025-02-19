#!/bin/bash

usage='Usage: -r <run number> -d <crab_subdirectory> -o <output_dir> -w <work_dir> -m <merge_dir>'

args=`getopt rdowm: -- "$@"`
if test $? != 0
     then
         echo $usage
         exit 1
fi

eval set -- "$args"
for i 
  do
  case "$i" in
      -r) shift; run_num=$2;shift;;
      -d) shift; crab_dir=$2;shift;;
      -o) shift; output_dir=$2;shift;;
      -w) shift; work_dir=$2;shift;;
      -m) shift; merge_dir=$2;shift;;
  esac      
done

#this_dir=`pwd`;
#echo "this is your working dir"
#echo `pwd`


echo "HELLO! " ${merge_dir}

if [ "X"${run_num} == "X" ]
    then
    echo "INVALID RUN NUMBER! Please give a valid run number!"
    echo $usage
    exit 
fi

if [ "X"${work_dir} == "X" ]
    then
#    work_dir=``;
    work_dir=`pwd`;
    echo " using default work dir" ${work_dir}
else
    echo " using work dir "${work_dir}
fi

if [ "X"${merge_dir} == "X" ]
    then
    merge_dir=${run_num}
    echo " using default merge dir" ${merge_dir}
else
    echo " using merge dir "${merge_dir}
fi

if [ "X"${crab_dir} == "X" ]
    then
    crab_dir=`\ls -rt1 ${work_dir}/${merge_dir} | grep "crab_" | tail -1 | awk '{print $NF}'`;
    echo " using default output dir" ${crab_dir}
else
    echo " using output dir "${crab_dir}
fi



echo 'Merging CRAB output ' ${run_num} 'crab_dir' ${crab_dir}

cd ${work_dir}/${merge_dir}/${crab_dir}/res;
#pwd;

# check root files
nroot=`\ls DQM_V0001_EcalPreshower_R*root | grep -vc ${run_num}`;
nmergeroot=`\ls DQM_V0001_EcalPreshower_R*root | grep -c ${run_num}`;
#echo $nmergeroot

if [ "${nroot}" == "0" ] && [ "${nmergeroot}" == "0" ]
then
    echo " NO root files" $nroot ".. exiting"
    exit
else
    echo " $nroot root files, $nmergeroot merged files"
fi

#rm -f EcalCosmicsHists_${run_num}.root
# now to hadd
hadd -f DQM_V0001_EcalPreshower_${run_num}.root DQM_V0001_EcalPreshower_R*root

# check log files
nstdout=`\ls -l CMSSW*stdout | wc | awk '{print $1}'`;
nmergeout=`\ls DQM_V0001_EcalPreshower_*log | grep -c ${run_num}`;
if [ "${nstdout}" == "0" ] #&& [ "${nmergeout}" == "0" ]
then
    echo " NO stdout files" $nstdout 
    #exit
else
    echo " $nstdout stdout files, $nmergeout merged files"
    cat CMSSW*stdout > DQM_V0001_EcalPreshower_${run_num}.log
fi

gzip -f DQM_V0001_EcalPreshower_${run_num}.log

if [ "X"${output_dir} == "X" ]
    then
    output_dir=/castor/cern.ch/user/c/ccecal/ES
    echo " use default output dir" $output_dir
    rfmkdir ${output_dir}
    rfchmod 775 ${output_dir}
    rfcp DQM_V0001_EcalPreshower_${run_num}.log.gz ${output_dir}
    rfcp DQM_V0001_EcalPreshower_${run_num}.root ${output_dir}
    rfchmod 775 ${output_dir}/DQM_V0001_EcalPreshower_${run_num}.log.gz
    rfchmod 775 ${output_dir}/DQM_V0001_EcalPreshower_${run_num}.root
else
    echo " using output dir "${output_dir}
    rfmkdir ${output_dir}
    rfchmod 775 ${output_dir}
    rfcp DQM_V0001_EcalPreshower_${run_num}.log.gz ${output_dir}
    rfcp DQM_V0001_EcalPreshower_${run_num}.root ${output_dir}
    rfchmod 775 ${output_dir}/DQM_V0001_EcalPreshower_${run_num}.log.gz
    rfchmod 775 ${output_dir}/DQM_V0001_EcalPreshower_${run_num}.root
#    rm -f EcalCosmicsHists_${run_num}.log
fi

size=`\ls -l DQM_V0001_EcalPreshower_${run_num}.root | grep -c ${run_num}`;
echo $size

if [ "${size}" == "0" ]
then
    echo " Warning: your merged file has size ZERO.. will not delete root files."
else
    echo
#    echo " Warning: deleting root files"
#    \ls EcalCosmicsHists_*root | grep -v ${run_num} | awk '{print "rm -f "$NF}' |sh
fi

#rm -rf CMSSW*stdout
#rm -rf CMSSW*stderr
#rm -rf crab*xml

#\ls -l;

