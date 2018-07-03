#!/bin/bash

COUNTER="0"

nf=$(cat ../createStep1.bash | awk '/nFiles/{i++}i==2' | cut -d= -f 2)
echo "$nf files to be sent to job queue"

do_files_exist(){
    fsExist=true 
    for ((index=1;index<=$1;index++))
    do
        if [ -e "error${index}.txt" ]; then
          :
        else
          fsExist=false
        fi
    done
    
}


# if data samples are larger switch to longer sleep times
rm error* output*
while [ $COUNTER -lt 15 ]; do
        bash ../createStep1.bash $COUNTER
        bash ../startStep1.bash
        
        while : 
        do
                fsExist=false
                sleep 1m
                do_files_exist $nf
                if [ $fsExist = true ];  then
                        break
                fi
        done 
        sleep 1m
        bash ../createStep2.bash $COUNTER
        bash ../startStep2.bash
        let COUNTER=COUNTER+1
        rm error* output*
done

bash ../createStep1.bash 15 True
bash ../startStep1.bash

while : 
do
        sleep 1m
        do_files_exist $nf
        if [ $fsExist = true ]; then
                break
        fi
done 

sleep 1m
bash ../createStep2.bash 15
bash ../startStep2.bash
