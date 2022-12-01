#!/bin/bash

if [ "$#" -ne 4 ]; then
  echo "Script for adding all root files from a folder locally or in EOS"
  echo "Usage of the script:"
  echo "$0 [doLocal] [inputFolderName] [outputFolderName] [baseName]"
  echo "doLocal = True: Merge local files. False: Merge files on CERN EOS"
  echo "inputFolderName = Name of the folder where the root files are"
  echo "outputFolderName = Name of the folder to which the output is transferred"
  echo "baseName = Name given for the output file without .root extension"
  exit
fi

LOCALRUN=$1
INPUTFOLDERNAME=${2%/}
OUTPUTFOLDERNAME=${3%/}
BASENAME=$4

if [ $LOCALRUN = true ]; then
  hadd -ff ${BASENAME}.root `ls ${INPUTFOLDERNAME}/*.root`
  mv ${BASENAME}.root $OUTPUTFOLDERNAME
else
  hadd -ff ${BASENAME}.root `xrdfs root://eoscms.cern.ch ls -u $INPUTFOLDERNAME | grep '\.root'`
  xrdcp ${BASENAME}.root root://eoscms.cern.ch/${OUTPUTFOLDERNAME}
  rm ${BASENAME}.root
fi
