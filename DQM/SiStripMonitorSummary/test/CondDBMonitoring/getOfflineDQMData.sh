#!/bin/bash

if [[ $4 == '' || $5 != '' ]]
then
  echo "This script accepts exactly 4 command line arguments"
  echo "Invoke it in this way:"
  echo "getOfflineDQMData.sh DBName DBAccount MonitoredVariable DBTag"
  echo "    DBname:            name of the database (Ex: cms_orcoff_prod)"
  echo "    DBAccount:         name of the database account (Ex: CMS_COND_31X_STRIP)"
  echo "    monitoredVariable: must be one among SiStripBadChannel, SiStripFedCabling, SiStripVoltage or RunInfo"
  echo "    DBTag:             name of the database tag (Ex: SiStripBadComponents_OfflineAnalysis_GR09_31X_v1_offline)"
  echo "Exiting."
  exit 1
fi

# The script accepts 4 command line parameters:
# Example: cms_orcoff_prod
export DBName=$1
# Example: CMS_COND_31X_STRIP
export DBAccount=$2
# Example: SiStripBadChannel
export monitoredVariable=$3
# Example: SiStripBadComponents_OfflineAnalysis_GR09_31X_v1_offline
export DBTag=$4

if [[ $monitoredVariable == "SiStripBadChannel" ]]
then
  baseDir=QualityLog
  baseName=QualityInfo_Run
elif [[ $monitoredVariable == "SiStripFedCabling" ]]
then
  baseDir=CablingLog
  baseName=QualityInfoFromCabling_Run
elif [[ $monitoredVariable == "SiStripVoltage" ]]
then
  baseDir=QualityLog
  baseName=QualityInfo_Run
elif [[ $monitoredVariable == "RunInfo" ]]
then
  baseDir=QualityLog
  baseName=QualityInfo_Run
else
  echo "The monitored variable that was entered is not valid!"
  echo "Valid choices are: SiStrip, SiStripFedCabling, SiStripVoltage or RunInfo."
  echo "Exiting."
  exit 1
fi

export workdir=/afs/cern.ch/cms/tracker/sistrcalib/WWW/CondDBMonitoring/$DBName/$DBAccount/DBTagCollection/$monitoredVariable/$DBTag/$baseDir

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

# Needed to run the root macro
export ARCH=slc4_ia32_gcc34
export ROOTSYS=/afs/cern.ch/sw/lcg/external/root/5.18.00/${ARCH}/root
export PATH=${PATH}:${ROOTSYS}/bin
export LD_LIBRARY_PATH=${ROOTSYS}/lib

if [[ -f $outFileTK ]]
then
  echo "ERROR: txt output file already exists. Exiting."
  exit 1;
fi

if [[ -f $outRootFileTK ]]
  then
  echo "ERROR: root output file already exists. Exiting."
  exit 1;
fi

# In order to make the for loop work properly, I insert underscores here, and transform them to spaces afterwards
for partName0 in Tracker TIB TID TOB TEC TIB_Layer_1_ TIB_Layer_2_ TIB_Layer_3_ TIB_Layer_4_ TID+_Disk_1_ TID+_Disk_2_ TID+_Disk_3_ TID-_Disk_1_ TID-_Disk_2_ TID-_Disk_3_ TOB_Layer_1_ TOB_Layer_2_ TOB_Layer_3_ TOB_Layer_4_ TOB_Layer_5_ TOB_Layer_6_ TEC+_Disk_1_ TEC+_Disk_2_ TEC+_Disk_3_ TEC+_Disk_4_ TEC+_Disk_5_ TEC+_Disk_6_ TEC+_Disk_7_ TEC+_Disk_8_ TEC+_Disk_9_ TEC-_Disk_1_ TEC-_Disk_2_ TEC-_Disk_3_ TEC-_Disk_4_ TEC-_Disk_5_ TEC-_Disk_6_ TEC-_Disk_7_ TEC-_Disk_8_ TEC-_Disk_9_ ;
do
  # Change underscores to spaces, and add a colon at the end
  partName=`echo $partName0 | sed s/_/\ /g | sed s/$/:/g`
  if [[ $partName0 == "Tracker" || $partName0 == "TIB" || $partName0 == "TID" || $partName0 == "TOB" || $partName0 == "TEC" ]] ;
  then
    subDetName=$partName0
    partType=""
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
      if [[ $fileName =~ "^${baseName}"  && $(( `wc -l "$workdir/$fileName" | awk '{print $1}'` - 51 )) > 0 ]] ; # File name must start with this string and must have at least that many lines
      then
        # Extract run number from first row of file
        runNumber=`head -n 1 "$workdir/$fileName" | awk '{print $NF}'`
        line=`head -n 52 $workdir/$fileName | tail -n 44 | grep "$partName" | awk -F ":" '{print $2}'`
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
    echo "Temporary file tmp.txt exists! remove it and retry!"
    exit 1
  fi
  # Remove last newline from file, otherwise the macro will read twice the last line
  cat $outFileTK | awk '{if(NR>1)print l;l=$0};END{if(NR>=1)printf("%s",l);}' > tmp.txt
  mv tmp.txt $outFileTK
  # Now we have the file with all the data for the given partName. Process it with the ROOT macro
  root -l -b -q makeTKTrend.cc\+\(\"$outFileTK\",\"$outRootFileTK\",\"$subDetName\",\"$partType\",$partNumber\)
  rm $outFileTK
done

# Run a macro that creates prettier plots
if [[ -f $outRootFileTK ]] ;
then
  root -l -b -q makePlots.cc+\(\"$outRootFileTK\",\"$outRootPlotsTK\"\)
fi
