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
#nsls /castor/cern.ch/cms/emuslice/2006/ > Runs.txt 
nsls /castor/cern.ch/user/b/boeriu/calibration_files > Runs.txt 
#takes all crosstalk runs
grep -e Crosstalk_ Runs.txt > AllCrosstalkRuns.txt 
#filters for only the main file from each run
grep -e ".raw" AllCrosstalkRuns.txt > GoodCrosstalkRuns_.txt
grep -e ".raw" GoodCrosstalkRuns_.txt > GoodCrosstalkRuns.txt

echo " ";
echo "will process runs: ";
cat GoodCrosstalkRuns.txt;
echo " ";

#for each file in GoodCrosstalkRuns, execute the following process
cat GoodCrosstalkRuns.txt | while read line 
do 
  #copy to /tmp/cscclaib directory
  #rfcp "/castor/cern.ch/cms/emuslice/2006/$line" "/tmp/csccalib/$line";
  rfcp "/castor/cern.ch/user/b/boeriu/calibration_files/$line" "/tmp/csccalib/$line";

  #create a config file using the perl script
  perl ConfigChanges.pl "/tmp/csccalib/$line";
  perl CreateConfigCrosstalk.pl "/tmp/csccalib/$line";
  #execute the the analyzer using the newly create config file. 
  # .root file will go into /tmp/csccalib
  cmsRun CSCxtalk.cfg;
  ls -l "/tmp/csccalib/";
  #remove .raw file from from /tmp/csccalib
  rm "/tmp/csccalib/$line" 
  #execute root macro. this will automatically take whatever file is in /tmp/csccalib
  #as it's input, so there should only be the new root file, nothing else. 
  #the root file will put it's image output into /afs/cern.ch/cms/CSC/html/csccalib/images/xxx
  #see the macro for more information. 
  root xTalkMacro.C;
  #clean out the /tmp/cscclaib directory. 
  rm /tmp/csccalib/csc*;
done

#clean out run files from current directory
rm Runs.txt
rm AllCrosstalkRuns.txt 
rm GoodCrosstalkRuns_.txt
rm GoodCrosstalkRuns.txt

#go into public and exec the perl script for web display
cd /afs/cern.ch/cms/CSC/html/csccalib
perl CreateTree_Items.pl 
cd -

