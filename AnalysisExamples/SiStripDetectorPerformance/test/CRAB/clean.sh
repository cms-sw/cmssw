#!/bin/sh

path=$1
value=0
[ "$2" != "" ] && value=$2  
for file in `ls -s -S $path | awk '{if($1==a) print $2 }' a=$value`;
  do
  rm -f $path/$file 
done
