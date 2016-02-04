#!/bin/bash

####################################################
#                                                  #
#Author:  Adam Roe, Northeastern University        #
#Contact: Adam.Roe@cern.ch, Oana.Boeriu@cern.ch    #
#Date:    November, 2006                           #
#                                                  #
####################################################

#this is for the "automated" processing of calibration runs. 
#it will copy a run from castor, process it using the Analyzer,
#run the root macro over it, and put it on the web. 
#the only input it needs is the files to run over. 

#enter " eval `scramv1 runtime -csh` " from src directory
#source exec.sh ### before processing

#Copies all runs from castor directory
nsls /castor/cern.ch/cms/emuslice/2006/ > Runs.txt 
#takes all crosstalk runs
grep -e PulseDAC Runs.txt > AllNoiseRuns.txt 
#filters for only the main file from each run
grep -e "RUI" AllNoiseRuns.txt > GoodNoiseRuns_.txt
#grep -e "csc_00000102_EmuRUI01_CFEB_PulseDAC_000.raw" GoodNoiseRuns_.txt > GoodNoiseRuns.txt
grep -e ".raw" GoodNoiseRuns_.txt > GoodNoiseRuns.txt

echo " ";
echo "will process runs: ";
cat GoodNoiseRuns.txt;
echo " ";

#for each file in GoodNoiseRuns, execute the following process
cat GoodNoiseRuns.txt | while read line 
do 
  #copy to /tmp/cscclaib directory
  rfcp "/castor/cern.ch/cms/emuslice/2006/$line" "/tmp/csccalib/$line";
  #create a config file using the perl script
  perl ConfigChanges.pl "/tmp/csccalib/$line";
  perl CreateConfigNoise.pl "/tmp/csccalib/$line";
  #execute the the analyzer using the newly create config file. 
  # .root file will go into /tmp/csccalib
  cmsRun CSCmatrix.cfg;
  ls -l "/tmp/csccalib/";
  #remove .raw file from from /tmp/csccalib
  #rm "/tmp/csccalib/$line";
  #execute root macro. this will automatically take whatever file is in /tmp/csccalib
  #as it's input, so there should only be the new root file, nothing else. 
  #the root file will put it's image output into /afs/cern.ch/cms/CSC/html/csccalib/images/xxx
  #see the macro for more information. 
  root noiseMatrixMacro.C;
  #clean out the /tmp/cscclaib directory. 
  rm /tmp/csccalib/csc*;
done

#clean out run files from current directory
rm Runs.txt
rm AllNoiseRuns.txt 
rm GoodNoiseRuns_.txt
rm GoodNoiseRuns.txt

#go into public and exec the perl script for web display
cd /afs/cern.ch/cms/CSC/html/csccalib
perl CreateTree_Items.pl 
cd -

