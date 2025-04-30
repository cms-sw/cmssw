#! /bin/bash

hipName="$(grep -m 1 "alignmentname=" $1 | cut -d= -f2)"

if [ -z "$hipName" ]
then
    echo "Value for 'alignmentname' not found in template. Please check your submission template."
else
    nohup ./$1 >> ../$hipName.log 2>&1 &
    echo $hipName $! >> ../pid.nohup
    echo "Please follow the log in '../$hipName.log'. To track progress live, use 'tail -f ../$hipName.log'."
    echo "The nohup job PID is appended to '../pid.nohup' in case the submission should be killed."
    echo "You can also use 'ps -ef | grep submit_' to find PIDs of currently running alignments."
fi

-- dummy change --
