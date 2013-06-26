#!/bin/bash

#nsls /castor/cern.ch/cms/emuslice/2006/ > Runs.txt 
ls /tmp/csccalib > Runs.txt
grep -e Saturation Runs.txt > AllSaturationRuns.txt 
grep -e "RUI" AllSaturationRuns.txt > GoodSaturationRuns.txt
grep -e "RUI" AllSaturationRuns.txt > GoodSaturationRunsDummy.txt

#Runs and RunsDummy begin as copies. while runs stays open, 
#the lines from RunsDummy are removed as they are processed. 
#grep -e "csc_00000508*" GoodSaturationRuns_.txt > GoodSaturationRuns.txt
#grep -e "csc_00000508*" GoodSaturationRuns_.txt > GoodSaturationRunsDummy.txt

echo " ";
echo "will process runs: ";
cat GoodSaturationRuns.txt;
echo " ";

#loop over ALL runs that you have grepped for. 
cat GoodSaturationRuns.txt | while read line  
do 
   #put into tempFile.txt all files from the same run. 
   grep -e "${line:9:4}" GoodSaturationRunsDummy.txt > tempFile.txt;
   #these two variables and if statment ensure that the next loop will
   #only run for a non-zero file size, i.e. if there are runs in tempFile.txt
   FILESIZE=$(stat -c%s tempFile.txt);
   MINSIZE=1
   if [ $FILESIZE -gt $MINSIZE ]; then
       #loop over all runs in tempFile.txt
#       cat tempFile.txt | while read line
#       do
	 #rfcp each file in tempFile.txt
	 #echo "copying $line from castor";
	 #rfcp "/castor/cern.ch/cms/emuslice/2006/$line" "/tmp/csccalib/$line";
	 #echo "copied $line from castor";
 #      done
       #Create a .cfg file, based on tempFile.txt. it will output CSCsaturation.cfg, which
       #will put all files from tempFile.txt in the .cfg for input to cmsRun. 
       #this will also rewrite GoodSaturationRunsDummy.txt to exclude the runs found in tempFile.txt,
       #which exlcudes multiple processing
       echo "creating config with perl";
       perl ConfigChanges.pl "/tmp/csccalib/$line";
       perl CreateConfigSaturation.pl; 
       echo "perl done, starting job";
       #run the cms job
       cmsRun CSCsaturation.cfg;
       #execute the root macro 
       #root saturationMacro.C
       #clean out the /tmp/csccalib directory 
       #rm /tmp/csccalib/csc* 
   else  
       echo "**** tempFile is empty  ****";
   fi 

done  

rm Runs.txt
rm AllSaturationRuns.txt 
rm GoodSaturationRuns.txt
rm GoodSaturationRunsDummy.txt
rm tempFile.txt

#cd /afs/cern.ch/cms/CSC/html/csccalib
#perl CreateTree_Items.pl 
#cd -
