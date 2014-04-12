#!/bin/bash
filename=`basename $1`
pathname=`dirname $1`

tag=`grep 'CREATIONTAG' $1 | head -n 1 | sed 's/.*>\(.*\)<.*/\1/'`

file_prefix=`echo $filename | sed 's/\(.*\)_[0-9]\+\.xml/\1/'`

file_list=`ls $pathname/*[0-9].xml`
echo ''
echo $file_list
echo ''
echo $file_list

