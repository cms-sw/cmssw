#!/bin/bash

caf_directory=$1 
castor_directory=/castor/cern.ch/cms/$1
label=`basename $1`
output_file=/tmp/friis/$label.root

echo $caf_directory
echo $castor_directory
echo $output_file

filelist=${label}_edmCopyPickMerge.list
rm -f $filelist
nsls $castor_directory | grep training | sed "s|^|$caf_directory/|" > $filelist

edmCopyPickMerge inputFiles_load=$filelist outputFile=$output_file maxSize=250000
