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
grep -e AFEB_DAC Runs.txt > AllAFEBRuns.txt;
#filters for only the main file from each run
grep -e "RUI" AllAFEBRuns.txt > GoodAFEBRuns_.txt
#grep -e ".raw" GoodAFEBRuns_.txt > GoodAFEBRuns.txt
grep -e "csc_00000515_EmuRUI00_AFEB_DAC_000.raw" GoodAFEBRuns_.txt > GoodAFEBRuns.txt
 
echo " ";
echo "will process runs: ";
cat GoodAFEBRuns.txt;
echo " ";
 
#for each file in GoodAFEBRuns, execute the following process
cat GoodAFEBRuns.txt | while read line
do
  #copy to /tmp/cscclaib directory
  rfcp "/castor/cern.ch/cms/emuslice/2006/$line" "/tmp/csccalib/$line";
  #create a config file using the perl script
  perl ConfigChanges.pl "/tmp/csccalib/$line";
  perl CreateConfigAFEB.pl "/tmp/csccalib/$line";
  #execute the the analyzer using the newly create config file.
  # .root file will go into /tmp/csccalib
  cmsRun CSCAFEBAnalysis.cfg
  ls -l "/tmp/csccalib/";
  #remove .raw file from from /tmp/csccalib
  #rm "/tmp/csccalib/$line";
  #execute root macro. this will automatically take whatever file is in /tmp/csccalib
  #as it's input, so there should only be the new root file, nothing else.
  #the root file will put it's image output into /afs/cern.ch/cms/CSC/html/csccalib/images/xxx
  #see the macro for more information.
  #root afebMacro.C
  #clean out the /tmp/cscclaib directory.
  #rm /tmp/csccalib/csc*;
done
 
#clean out run files from current directory
rm Runs.txt
rm AllAFEBRuns.txt
rm GoodAFEBRuns_.txt
rm GoodAFEBRuns.txt
 
#go into public and exec the perl script for web display
cd /afs/cern.ch/cms/CSC/html/csccalib
perl CreateTree_Items.pl
cd -
