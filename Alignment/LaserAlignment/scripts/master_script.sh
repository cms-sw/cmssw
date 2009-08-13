#!/bin/sh

# define needed variables 
num_files=44
run=110452
events=( 1321046 1401963 1483282 1563969 1644524 1725419 1967629 2048560 2129353 2210845 2291602 2372518 3049348 3130348 3211861 3294883 3376226 3457577 3538316 3619788 3700609 3781431 3862678 3943760)
len_ev=${#events[*]}

file_name1=t0ProducerStandalone_part3_
file_name2=write_python_config_files_LAS_
file_name3=run_job_LAS_
file_name4=write_bjobs_LAS_
file_name5=run_on_LSF_LAS_
file_name6=merge_
file_ext1=RunXXXXXX_EvYYYYYY
file_ext2=.py
file_ext3=.sh
file_ext4=.log

#t0ProducerStandalone_part3_RunXXXXXX_EvYYYYYY.py
#write_python_config_files_LAS_RunXXXXXX_EvYYYYYY.sh
#run_job_LAS_RunXXXXXX_EvYYYYYY.sh
#write_bjobs_LAS_RunXXXXXX_EvYYYYYY.sh
#run_on_LSF_LAS_RunXXXXXX_EvYYYYYY.sh
#merge_RunXXXXXX_EvYYYYYY.py

# write the merge_RunXXXXXX_EvYYYYYY.py file
./write_merge_RunXXXXXX_EvYYYYYY.sh `echo $num_files`

iev_last=`expr \( $len_ev - 1 \)`
ev_last=`expr \( ${events[$iev_last]} + 150000 \)`

echo "Write global shell and python scripts..."
# write the rest of the shell and python scripts
for (( i = 0;  i < $len_ev; ++i ))
    do
        file_extN=`echo "Run$run""_Ev${events[$i]}"`
        cp $file_name1$file_ext1$file_ext2 $file_name1$file_extN$file_ext2
        cp $file_name2$file_ext1$file_ext3 $file_name2$file_extN$file_ext3 
        cp $file_name3$file_ext1$file_ext3 $file_name3$file_extN$file_ext3
        cp $file_name4$file_ext1$file_ext3 $file_name4$file_extN$file_ext3 
        cp $file_name5$file_ext1$file_ext3 $file_name5$file_extN$file_ext3 
        cp $file_name6$file_ext1$file_ext2 $file_name6$file_extN$file_ext2 

        sed -i "s/XXXXXX/$run/g" $file_name1$file_extN$file_ext2
        sed -i "s/YYYYYY/${events[$i]}/g" $file_name1$file_extN$file_ext2

        sed -i "s/XXXXXX/$run/g" $file_name2$file_extN$file_ext3
        sed -i "s/YYYYYY/${events[$i]}/g" $file_name2$file_extN$file_ext3
        sed -i "s/nFiles/$num_files/g" $file_name2$file_extN$file_ext3

        sed -i "s/XXXXXX/$run/g" $file_name3$file_extN$file_ext3
        sed -i "s/YYYYYY/${events[$i]}/g" $file_name3$file_extN$file_ext3

        sed -i "s/XXXXXX/$run/g" $file_name4$file_extN$file_ext3
        sed -i "s/YYYYYY/${events[$i]}/g" $file_name4$file_extN$file_ext3
        sed -i "s/nFiles/$num_files/g" $file_name4$file_extN$file_ext3
 
        sed -i "s/XXXXXX/$run/g" $file_name5$file_extN$file_ext3
        sed -i "s/YYYYYY/${events[$i]}/g" $file_name5$file_extN$file_ext3
        sed -i "s/nFiles/$num_files/g" $file_name5$file_extN$file_ext3

        sed -i "s/XXXXXX/$run/g" $file_name6$file_extN$file_ext2
        sed -i "s/YYYYYY/${events[$i]}/g" $file_name6$file_extN$file_ext2

        if (( i == $iev_last ))
        then
            sed -i "s/000000/$ev_last/g" $file_name1$file_extN$file_ext2
            sed -i "s/000000/$ev_last/g" $file_name6$file_extN$file_ext2
        else
            sed -i "s/000000/${events[$i+1]}/g" $file_name1$file_extN$file_ext2
            sed -i "s/000000/${events[$i+1]}/g" $file_name6$file_extN$file_ext2
        fi
        
    done
echo "global shell and python scripts are done. Next step..."
echo "Write t0ProducerStandalone_RunXXXXXX_EvYYYYYY_Fi.py scripts..."
for (( i = 0;  i < $len_ev; ++i ))
    do
    file_extN=`echo "Run$run""_Ev${events[$i]}"`
    ./$file_name2$file_extN$file_ext3
    done
echo "t0ProducerStandalone_RunXXXXXX_EvYYYYYY_Fi.py scripts are done. Next step..."
echo "Write write_bjobs_LAS_RunXXXXXX_EvYYYYYY_Fi.sh scripts..."
for (( i = 0;  i < $len_ev; ++i ))
    do
    file_extN=`echo "Run$run""_Ev${events[$i]}"`
    ./$file_name4$file_extN$file_ext3
    done
echo "write_bjobs_LAS_RunXXXXXX_EvYYYYYY_Fi.sh scripts are done. Next step..."
echo "Create directories to store the output of LSF batch jobs..."
echo "Warning, this depends on lxplus machine and user!!!"
for (( i = 0;  i < $len_ev; ++i ))
    do
    file_extN=`echo "Run$run""_Ev${events[$i]}"`
    mkdir -p /tmp/aperiean/Run$run/$file_extN
    done
echo "directories to store the output of LSF batch job were created. Next step..."
echo "Submit LAS jobs on LSF batch system..."
for (( i = 0;  i < $len_ev; ++i ))
    do
    file_extN=`echo "Run$run""_Ev${events[$i]}"`
    ./$file_name5$file_extN$file_ext3
    #ls $file_name5$file_extN$file_ext3
    done
echo "LAS jobs on LSF batch system were submited. Next step..."
# check job status
bjobs >& check
bjobs_check=`less check`
rm check
while [ "$bjobs_check" != "No unfinished job found" ]
    do
    sleep 60
    du -h /tmp/aperiean/Run$run/
    date '+%T'
    # check job status every 60 seconds
    bjobs >& check
    bjobs_check=`less check`
    rm check
    done
if [ "$bjobs_check" = "No unfinished job found" ]
then
    echo "Run merge_RunXXXXXX_EvYYYYYY.py files..."
    for (( i = 0;  i < $len_ev; ++i ))
	do
	file_extN=`echo "Run$run""_Ev${events[$i]}"`
	cmsRun $file_name6$file_extN$file_ext2 >& $file_name6$file_extN$file_ext4 
	#ls $file_name6$file_extN$file_ext2
	done    
    echo "files TkAlLAS_RunXXXXXX_EvYYYYYY.root were done."
    echo "They can be found at lxplus218:/tmp/aperiean/Run$run."
    echo "The End"
fi
