#!/bin/csh

cmsenv

echo "Starting Pedestal job part 2"

# this parameters are ajusted from send_pedestalProducePayloads.csh script
set tagName = <send_tag>
@ beginRunNumber = <send_run>
set threshold = <send_thresh>

echo "Dumping the conditions for run: $beginRunNumber"
./makedump.csh  Pedestals $tagName $beginRunNumber
cp DumpPedestals_Run${beginRunNumber}.txt dump.txt

# removing test.db if existed
if (-e test.db) then
 rm test.db
endif

echo "Unzipping the pedestals: \n"
rm *-peds_ADC_*.txt
unzip pedstxt.zip

set listOfFiles = "listOfPeds.txt"
# removing listof peds file if existed
if (-e $listOfFiles) then
 rm ${listOfFiles}
endif

ls *-peds_ADC_*.txt > $listOfFiles
echo "The filenames are saved in: " ${listOfFiles}

set runsWithChanges = ()

echo "\n --Looping over the raw peds files --"
foreach line (`cat ${listOfFiles}`)
    # find a substring at position 1 with length 6 - runnumber 
   set stringRun =  "`expr substr $line 1 6`"

   @ currentRun = $stringRun # convert to integer
   echo "$currentRun  $line" 

     cp template_chancheck.py chancheck_${currentRun}.py				
     sed -i "s:<run>:${currentRun}:g" chancheck_${currentRun}.py
     sed -i "s:<threshold>:${threshold}:g" chancheck_${currentRun}.py
     sed -i "s:<filename>:${line}:g" chancheck_${currentRun}.py

    set diffFile1 = "${currentRun}_diffs1.txt"
    set diffFile2 = "${currentRun}_diffs2.txt"
    set diffFile3 = "${currentRun}_diffs3.txt"

    if (-e $diffFile1) then
    rm $diffFile1
    endif
    if (-e $diffFile2) then
    rm $diffFile2
    endif 
    if (-e $diffFile3) then
    rm $diffFile3
    endif

    cp dump.txt dump0.txt
 #The chancheck job will overwrite dump.txt. Therefore I cp it to dump0.txt for further comparisons
    sed -i "s:<filename>:${line}:g" chancheck_${currentRun}.py
    cmsRun chancheck_${currentRun}.py > ${diffFile1}
    cp dump.txt ${currentRun}_merged.txt 
 
#dump0.txt ${currentRun}_merged.txt

    echo "\n Compiling a cpp utility to make another diff check" 
    g++ diffpedestals.cpp
    echo "     Compiled\n"
    ./a.out dump0.txt ${currentRun}_merged.txt ${threshold} > ${diffFile2}
    
   diff dump0.txt ${currentRun}_merged.txt > ${diffFile3}
    echo "\nFirst diff:  `wc -l ${diffFile1}` second diff: " `wc -l ${diffFile2}`
  
# get the numbeer of lines in diffs
    set nlines1 = `wc -l ${diffFile1} | awk '{print $1'}` 
    set nlines2 = `wc -l ${diffFile2} | awk '{print $1'}`
    #echo $nlines1 $nlines2
       if (${nlines1} != {$nlines2}) then
        echo " \n  +++++++ Warning!! Diffs don't match ++++++++++++++"
       endif

   if (${nlines1}>1) then
    set runsWithChanges = ($runsWithChanges ${currentRun})
  echo "\n This conditions seem to have some diferences. Lets put them in sqlite file, test.db"
    ./write.csh Pedestals CalibCalorimetry/HcalStandardModules/test/${currentRun}_merged.txt  $tagName $currentRun
    endif

end  #end of list of peds
echo "-- end the loop --"

echo "the list of runs with changes. Threshold: ${threshold}\n ${runsWithChanges}\n"
echo "First run with changes: $runsWithChanges[1]"

echo "\n Createing the metadata file for DropBox.. "
cp template_metadata.txt metadata.txt
sed -i "s:<tag>:${tagName}:g" metadata.txt
sed -i "s:<run>:${runsWithChanges[1]}:g" metadata.txt
echo "\t\t Medatada file has been created. \n"

echo "Now you need to: \
       1) double check that test.db file produced is valid. (dump the conditions, see number of lines, compare against dump0.txt)\
       2) Double chack that metadata.txt file is valid: tag name and run number\
       3) Load the file to Orcon data base. wGet the dropbox script and execute:\
wget http://condb.web.cern.ch/condb/DropBoxOffline/dropBoxOffline.sh \
/bin/sh dropBoxOffline.sh test.db metadata.txt \
\n Once loaded, please, update the documentation: https://twiki.cern.ch/twiki/bin/viewauth/CMS/HcalPedestalsTags2011\n"
