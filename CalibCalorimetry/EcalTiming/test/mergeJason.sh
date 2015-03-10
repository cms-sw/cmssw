#!/bin/bash

usage='Usage: -r <run number> -d <crab_subdirectory> -o <output_dir>'

args=`getopt rd: -- "$@"`
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
      -at) shift; analy_type=$2;shift;;
  esac      
done



if [ "X"${analy_type} == "X" ]
    then
    analy_type="Laser"
    echo " using default analysis type of Laser"
else
    echo " doing analysis on ${analy_type} of events  "
fi

if [ "X"${run_num} == "X" ]
    then
    echo "INVALID RUN NUMBER! Please give a valid run number!"
    echo $usage
    exit 
fi


if [ "X"${crab_dir} == "X" ]
    then
    crab_dir=`\ls -rt1 ${analy_type}_${run_num} | grep "crab_" | tail -1 | awk '{print $NF}'`;
#    echo " using default output dir" ${crab_dir}
#else
#    echo " using output dir "${crab_dir}
fi

echo 'Merging CRAB output' ${run_num} 'crab_dir' ${crab_dir}

cd ${analy_type}_${run_num}/${crab_dir}/res;
#pwd;

# check root files
nroot=`\ls ${analy_type}_*root | grep -vc ${run_num}`;
nmergeroot=`\ls ${analy_type}_*root | grep -c ${run_num}`;
#echo $nmergeroot

if [ "${nroot}" == "0" ] && [ "${nmergeroot}" == "0" ]
then
    echo " NO root files" $nroot ".. exiting"
    exit
else
    echo " $nroot root files, $nmergeroot merged files"
fi

#rm -f ${analy_type}_${run_num}.root
# now to hadd
hadd -f ${analy_type}_${run_num}.root ${analy_type}_*root

if [ "${analy_type}" == "Laser" ]
then
   echo "Running the laser combination analysis"
   CheckEcalTiming.sh -p Run_${run_num} -ff True -ffn ${analy_type}_${run_num}.root -n 25 -t 25 -dt Laser -l 2
   ####rm -fr data
   ####rm -fr conf
   mv -f log/Timing${analy_type}_Run_${run_num}.*.root ${analy_type}_${run_num}.root
   ####rm -fr log
fi

if [ "${analy_type}" == "Timing" ]
then
   echo "Running the Timing combination analysis"
   #CheckEcalTiming.sh -p Run_${run_num} -ff True -ffn ${analy_type}_${run_num}.root -n 5 -t 25 -dt Physics -l 2 -aa 5.0 -as 0.0 -dr True 
   #######################CheckEcalTiming.sh -p Run_${run_num} -ff True -ffn ${analy_type}_${run_num}.root -n 0 -t 15 -dt Physics -l 2 -aa 5.0 -as 0.0 -dr True -tt True
   ###rm -fr data
   ###rm -fr conf
   #######################mv -f log/TimingPhysics_Run_${run_num}.*.root ${analy_type}_${run_num}.root
   echo "done"
   ###rm -fr log
fi


# check log files
nstdout=`\ls -l CMSSW*stdout | wc | awk '{print $1}'`;
nmergeout=`\ls ${analy_type}_*log | grep -c ${run_num}`;
if [ "${nstdout}" == "0" ] #&& [ "${nmergeout}" == "0" ]
then
    echo " NO stdout files" $nstdout 
    #exit
else
    echo " $nstdout stdout files, $nmergeout merged files"
    cat CMSSW*stdout > ${analy_type}_${run_num}.log
fi

if [ "X"${output_dir} == "X" ]
    then
    output_dir=/castor/cern.ch/user/c/ccecal/${analy_type}
    echo " use default output dir" $output_dir
    rfmkdir ${output_dir}
    rfchmod 775 ${output_dir}
    rfcp ${analy_type}_${run_num}.log ${output_dir}
    rfcp ${analy_type}_${run_num}.root ${output_dir}
    rfchmod 775 ${output_dir}/${analy_type}_${run_num}.log
    rfchmod 775 ${output_dir}/${analy_type}_${run_num}.root
else
    echo " using output dir "${output_dir}
    rfmkdir ${output_dir}
    rfchmod 775 ${output_dir}
    rfcp ${analy_type}_${run_num}.log ${output_dir}
    rfcp ${analy_type}_${run_num}.root ${output_dir}
    rfchmod 775 ${output_dir}/${analy_type}_${run_num}.log
    rfchmod 775 ${output_dir}/${analy_type}_${run_num}.root
    ####rm -f ${analy_type}_${run_num}.log
fi

size=`\ls -l ${analy_type}_${run_num}.root | grep -c ${run_num}`;
echo $size

if [ "${size}" == "0" ]
then
    echo " Warning: your merged file has size ZERO.. will not delete root files."
else
    echo
#    echo " Warning: deleting root files"
#    \ls ${analy_type}_*root | grep -v ${run_num} | awk '{print "rm -f "$NF}' |sh
fi

#rm -rf CMSSW*stdout
#rm -rf CMSSW*stderr
#rm -rf crab*xml

#\ls -l;

