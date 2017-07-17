#!/bin/bash

DATASET=$1
FILELIST=$2

if [ "$#" -lt '1' ] || [ $1 == "-h" ] || [ $1 == "--help" ]
then
  printf "\nUsage: "
  printf "downloadFiles.sh [DataSet] [FileName(optional)]\n\n"
  exit 1;
fi
 
if [ "$#" != '2' ]
then
FILELIST="fileList.txt"
fi

das_client.py --limit=0 --query="file dataset=$DATASET | grep file.name, file.nevents > 0" >& $FILELIST


