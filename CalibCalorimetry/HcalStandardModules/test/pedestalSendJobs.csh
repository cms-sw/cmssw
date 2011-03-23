#!/bin/csh

echo "---- Starting Pedestal job. Part 1 --"

#set datasetName = "/TestEnables/Commissioning11-v3/RAW"
set datasetName = "/TestEnables/Commissioning11-v1/RAW"
#set datasetName =  "/TestEnables/HIRun2010-v1/RAW"
#set datasetName = "/TestEnables/Run2010B-v1/RAW"
@ beginRunNumber = 159769
@ endRunNumber = 999999

set castorDir = "/castor/cern.ch/user/a/andrey/peds2011/"

echo "Dataset is" ${datasetName}

## The run number is coded in the files e.g. ../157/803/...
# Get the position of the first digit:  
  @ pos = `expr length $datasetName` + 17
# Get the position of the 4th digit:  
  @ pos2 = $pos + 4 
#echo $pos $pos2

set listOfFiles = "listOfFiles.txt"
set currentDir = `pwd`
echo 'We are in: ' ${currentDir}

echo 'Looking Up DBS for the files in ' $datasetName
# if the file exists, recreate it

if (-e $listOfFiles) then
 rm *${listOfFiles}
endif
 touch $listOfFiles


if (-e temp_listOfFiles.txt) rm temp_listOfFiles.txt
dbsql "find file where dataset=${datasetName}" > temp_listOfFiles.txt

set nlines = `cat temp_listOfFiles.txt | wc -l`


echo  "Create the list of files to be processed between $beginRunNumber and $endRunNumber runs"
@ currentRun = 111111
@ tempRun = 100000

foreach line (`cat temp_listOfFiles.txt`)
# Check if the line contains the path (/store) - this is a filename
@ aa = `expr index $line "/store"`
  if ($aa == 1 )  then 
	## Extracting the run number from the name of the files e.g. ../157/803/...
	set stringRun =  "`expr substr $line $pos 3``expr substr $line $pos2 3`" # find a substring at position pos with length 
	@ currentRun = $stringRun # convert to integer
#	echo "$currentRun  $endRunNumber"

	#Want only the runs after endRunNumber processed
        if ($currentRun >= $beginRunNumber && $currentRun <= $endRunNumber) then
	    echo $line >> $listOfFiles
#	    echo $line
#	    echo "$currentRun  $tempRun"
	    #Want to separate the files corresponding to the same run
	    if ($currentRun != $tempRun) then

		cp template_batch_job.csh batch_job_${tempRun}.csh				
		sed -i "s:<dir>:${currentDir}:g" batch_job_${tempRun}.csh
		sed -i "s:<job>:batch_run_${tempRun}.py:g" batch_job_${tempRun}.csh
		sed -i "s:<run>:${tempRun}:g" batch_job_${tempRun}.csh
		sed -i "s:<castor>:${castorDir}:g" batch_job_${tempRun}.csh
		echo "Sending the job to cmscaf: batch_run_${tempRun}.py"
		bsub -q cmscaf1nd -J job1 < batch_job_${tempRun}.csh 
	    
		rm batch_job_${tempRun}.csh

		# Create a new job (run number is changed)
		echo "Creating a batch job for run: $currentRun"
		cp template_batch_run.py batch_run_${currentRun}.py
		sed -i "s:#INPUTFILES:#INPUTFILES\n'${line}',:g" batch_run_${currentRun}.py
		
	    else 
		    # Add the filename to the batch job
		sed -i "s:#INPUTFILES:#INPUTFILES\n'${line}',:g" batch_run_${currentRun}.py
            endif
	    @ tempRun = $currentRun
	else if ($currentRun < $beginRunNumber) then
                      break
	endif
    endif
end  #end of list from  sqdb 


echo "The filenames are saved in: " ${listOfFiles}
rm temp_listOfFiles.txt

