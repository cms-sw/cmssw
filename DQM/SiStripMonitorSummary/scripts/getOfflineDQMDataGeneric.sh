#!/bin/bash

if [[ $3 == '' || $4 != '' ]]
then
  echo "This script accepts exactly 3 command line arguments"
  echo "Invoke it in this way:"
  echo "getOfflineDQMData.sh Path dirName fileNameTemplate"
  echo "    Path:            name of the path where the files are"
  echo "    dirName: name of the directory where the log files are. Usually QualityLog"
  echo "    fileNameTemplate: prefix of the log file name"
  echo "Exiting."
  exit 1
fi

# The script accepts 4 command line parameters:
# Example: cms_orcoff_prod
export Path=$1
export baseDir=$2
export baseName=$3

export workdir=$Path/$baseDir

if [[ !( -d $workdir ) ]]
then
  echo "Directory $workdir does not exist!"
  echo "Exiting."
  exit 1
fi

# Name of file to be skipped from loop (contains no real data)
export badFileName=${baseName}1.txt

# [Temporary] text file with summary data for each iteration step
export outFileTK=./TrackerSummary.txt

# File with output histograms
export outRootFileTK=./TrackerSummary.root

# File with prettier plots
export outRootPlotsTK=./TrackerPlots.root

#if [[ -f $outFileTK ]]
#then
#  echo "ERROR: txt output file already exists. Exiting."
#  exit 1;
#fi

#if [[ -f $outRootFileTK ]]
#  then
#  echo "ERROR: root output file already exists. Exiting."
#  exit 1;
#fi

# Use the root installation defined above
#source ${ROOTSYS}/bin/thisroot.sh

# In order to make the for loop work properly, I insert underscores here, and transform them to spaces afterwards
for partName0 in Tracker TIB TID TOB TEC TIB_Layer_1_ TIB_Layer_2_ TIB_Layer_3_ TIB_Layer_4_ TID+_Disk_1_ TID+_Disk_2_ TID+_Disk_3_ TID-_Disk_1_ TID-_Disk_2_ TID-_Disk_3_ TOB_Layer_1_ TOB_Layer_2_ TOB_Layer_3_ TOB_Layer_4_ TOB_Layer_5_ TOB_Layer_6_ TEC+_Disk_1_ TEC+_Disk_2_ TEC+_Disk_3_ TEC+_Disk_4_ TEC+_Disk_5_ TEC+_Disk_6_ TEC+_Disk_7_ TEC+_Disk_8_ TEC+_Disk_9_ TEC-_Disk_1_ TEC-_Disk_2_ TEC-_Disk_3_ TEC-_Disk_4_ TEC-_Disk_5_ TEC-_Disk_6_ TEC-_Disk_7_ TEC-_Disk_8_ TEC-_Disk_9_ ;
do
  # Change underscores to spaces, and add a colon at the end
  partName=`echo $partName0 | sed s/_/\ /g | sed s/$/:/g`
  if [[ $partName0 == "Tracker" || $partName0 == "TIB" || $partName0 == "TID" || $partName0 == "TOB" || $partName0 == "TEC" ]] ;
  then
    subDetName=$partName0
    partType="All"
    partNumber=0
  else
    subDetName=`echo $partName | awk '{print $1}'`
    partType=`echo $partName | awk '{print $2}'`
    partNumber=`echo $partName | awk '{print $3}'`
  fi
  touch $outFileTK
  echo Processing subDetName $subDetName, partType $partType, partNumber $partNumber
 
  for fileName in `ls $workdir` ;
  do
    # This has not any real data
    if [[ $fileName != "${baseName}1.txt" ]] ;
    then
      if [[ $fileName == "${baseName}"*  && $(( `wc -l "$workdir/$fileName" | awk '{print $1}'` - 51 )) > 0 ]] ; # File name must start with this string and must have at least that many lines
      then
        # Extract run number from first row of file
# AV runNumber of line definition changed to be less dependent on the details of the log file
#        runNumber=`head -n 1 "$workdir/$fileName" | awk '{print $6}'`
#        line=`head -n 52 $workdir/$fileName | tail -n 44 | grep "$partName" | awk -F ":" '{print $2}'`
        runNumber=`grep "New IOV" "$workdir/$fileName" | awk '{print $6}'`
        line=`grep -A 10000 "Global Info" $workdir/$fileName | grep -B 10000 Detid | grep "$partName" | awk -F ":" '{print $2}'`
#                echo $line
#                echo $fileName
        nBadModulesTK=`echo $line | awk {'print $1'}`
        #         # If we have bad modules, check how many fibers they have
        #         if [[ $nBadModulesTK > '0' ]] ;
        #         then
        #           nBadModulesWithThreeFibersTK=0;
        #           for fiber in `tail +57 $workdir/$fileName  | grep -v TIB | grep -v TID | grep -v TOB | grep -v TEC | awk 'NF>0' | awk '$2>0 {print $4}'` ;
        #           do
        #             if [[ $fiber != 'x' ]] ;
        #             then
        #               let nBadModulesWithThreeFibersTK++;
        #             fi
        #           done
        #           let nBadModulesWithTwoFibersTK=$nBadModulesTK-$nBadModulesWithThreeFibersTK
        #           let nBadFibersFromModulesTK=$nBadModulesWithThreeFibersTK*3+$nBadModulesWithTwoFibersTK*2
        #           nAllBadFibersTK=`echo $line | awk {'print $3'}`
        #           let nBadFibersTK=$nAllBadFibersTK-$nBadFibersFromModulesTK
        #         else
        #           # We don't have bad modules: all bad fibers are tagged individually
        #           nBadFibersTK=`echo $line | awk {'print $3'}`
        #           nAllBadFibersTK=$nBadFibersTK
        #         fi
        nAllBadFibersTK=`echo $line | awk '{print $2}'`
        nAllBadAPVsTK=`echo $line | awk '{print $3}'`
        let nBadAPVsFromFibersTK=$nAllBadFibersTK*2
        let nBadAPVsTK=$nAllBadAPVsTK-$nBadAPVsFromFibersTK
        nAllBadStripsTK=`echo $line | awk '{print $4}'`
        let nBadStripsFromAPVsTK=$nAllBadAPVsTK*128;
        let nBadStripsTK=$nAllBadStripsTK-$nBadStripsFromAPVsTK
        echo $runNumber $nBadModulesTK $nAllBadFibersTK $nAllBadAPVsTK $nBadStripsTK $nBadStripsFromAPVsTK $nAllBadStripsTK >> $outFileTK
      elif [[ $fileName =~ "^${baseName}"  && $(( 52 - `wc -l "$workdir/$fileName" | awk '{print $1}'` )) > 0 ]] ;
      then
        echo "*** ERROR! Skipping file: $fileName because it is blank or has too few lines!"
        echo "*** Number of lines in the file =" `wc -l "$workdir/$fileName" | awk '{print $1}'`
        echo "*** The execution will continue without that IOV."
      fi
    fi
  done
  if [[ -f tmp.txt ]] ;
  then
#    echo "Temporary file tmp.txt exists! remove it and retry!"
#    exit 1
    rm tmp.txt;
  fi
  # Remove last newline from file, otherwise the macro will read twice the last line
  cat $outFileTK | awk '{if(NR>1)print l;l=$0};END{if(NR>=1)printf("%s",l);}' > tmp.txt
  mv tmp.txt $outFileTK
  # Now we have the file with all the data for the given partName. Process it with the ROOT macro
  makeTKTrend $outFileTK $outRootFileTK $subDetName $partType $partNumber
  rm $outFileTK
done

# Run a macro that creates prettier plots
if [[ -f $outRootFileTK ]] ;
then
    makePlots $outRootFileTK $outRootPlotsTK
fi
