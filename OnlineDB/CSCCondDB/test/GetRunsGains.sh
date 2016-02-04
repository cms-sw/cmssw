#!/bin/bash


#//nsls /castor/cern.ch/cms/emuslice/2006/ > Runs.txt 
nsls /castor/cern.ch/user/b/boeriu/calibration_files > Runs.txt 
grep -e Gain Runs.txt > AllGainsRuns.txt 
grep -e "RUI01" AllGainsRuns.txt > GainsRuns.txt
grep -e "30" GainsRuns.txt > GoodGainsRuns.txt
#cp GoodGainsRuns.txt GoodGainsRunsDummy.txt
#grep -e "29" GainsRuns.txt > GoodGainsRunsDummy.txt

#Runs and RunsDummy begin as copies. while runs stays open, 
#the lines from RunsDummy are removed as they are processed. 


echo " ";
echo "will process runs: ";
cat GoodGainsRuns.txt;
echo " ";

#loop over ALL runs that you have grepped for. 
cat GoodGainsRuns.txt | while read line  
do 
   #put into tempFile.txt all files from the same run. 
   grep -e "${line:9:4}" GoodGainsRuns.txt > tempFile.txt;
   #these two variables and if statment ensure that the next loop will
   #only run for a non-zero file size, i.e. if there are runs in tempFile.txt
   FILESIZE=$(stat -c%s tempFile.txt);
   MINSIZE=1
   if [ $FILESIZE -gt $MINSIZE ]; then
       #loop over all runs in tempFile.txt
       cat tempFile.txt | while read line
       do
	 #rfcp each file in tempFile.txt
#	 rfcp "/castor/cern.ch/cms/emuslice/2006/$line" "/tmp/csccalib/$line";
	 rfcp "/castor/cern.ch/user/b/boeriu/calibration_files/$line" "/tmp/csccalib/$line";
	 echo "$line";
       done
       #Create a .cfg file, based on tempFile.txt. it will output CSCgain.cfg, which
       #will put all files from tempFile.txt in the .cfg for input to cmsRun. 
       #this will also rewrite GoodGainsRunsDummy.txt to exclude the runs found in tempFile.txt,
       #which exlcudes multiple processing
       #echo "creating config with perl";
       perl ConfigChanges.pl "/tmp/csccalib/$line";
       perl CreateConfigGains.pl;
       echo "perl done, running job";
       #run the cms job
       cmsRun CSCgain.cfg;
       #execute the root macro, first get rid of the .dat file so the macro find the .root file
       #rm /tmp/csccalib/*.dat
       root gainsMacro.C
       #clean out the /tmp/csccalib directory 
       #rm /tmp/csccalib/csc* 
   else  
       echo "**** tempFile is empty  ****"
   fi

done 

rm Runs.txt
rm AllGainsRuns.txt 
rm GainsRuns.txt 
rm GoodGainsRuns.txt
rm GoodGainsRunsDummy.txt
rm tempFile.txt

cd /afs/cern.ch/cms/CSC/html/csccalib
perl CreateTree_Items.pl 
cd -
