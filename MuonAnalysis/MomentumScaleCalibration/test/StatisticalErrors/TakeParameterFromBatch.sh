#!/bin/bash

if [ $# -lt 1 ]
then
     echo "Error, usage is:"
     echo "$0 N (dir)"
     echo "Where"
     echo "  - N is the number of the line at which the parameter to extract is"
     echo "  - dir is the directory where to look for FitParameters_xxx.txt file"
     echo "    (optional, default is the last created)"
     exit
fi

if [ $# -eq 2 ]
then
    dir=$2
else
    dir=`\ls -d1rt StatErrors_* | tail -1`
fi

#echo $dir

echo Taking parameter at line $1

if [ -f Values.txt ]
then
    rm Values.txt
fi

for file in $(ls ${dir}/FitParameters_*.txt)
do
    value=$(sed -n "${1}p" ${file} | awk '{print $9}' | awk -F+ '{print $1}')
    #echo "dir = $dir, file = $file value = $value"
    echo $value >> Values.txt
done
