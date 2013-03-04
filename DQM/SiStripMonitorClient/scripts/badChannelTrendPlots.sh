#!/bin/bash

# The script accepts 4 command line parameters:
# Example: Data2012/Beam
export TkMapDir=$1
# Example: MinimumBias
export Dataset=$2

homedir=`pwd`
echo $homedir

baseName=PCLBadComponents.log

export workdir=/data/users/event_display/$TkMapDir

if [[ !( -d $workdir ) ]]
then
  echo "Directory $workdir does not exist!"
  echo "Exiting."
  exit 1
fi

# [Temporary] text file with summary data for each iteration step
export outFileTK=./TrackerSummary.txt

# File with output histograms
export outRootFileTK=./TrackerSummary.root

# File with prettier plots
export outRootPlotsTK=./TrackerPlots.root

# In order to make the for loop work properly, I insert underscores here, and transform them to spaces afterwards
for partName0 in Tracker TIB TID TOB TEC TIB_Layer_1_ TIB_Layer_2_ TIB_Layer_3_ TIB_Layer_4_ TID+_Disk_1_ TID+_Disk_2_ TID+_Disk_3_ TID-_Disk_1_ TID-_Disk_2_ TID-_Disk_3_ TOB_Layer_1_ TOB_Layer_2_ TOB_Layer_3_ TOB_Layer_4_ TOB_Layer_5_ TOB_Layer_6_ TEC+_Disk_1_ TEC+_Disk_2_ TEC+_Disk_3_ TEC+_Disk_4_ TEC+_Disk_5_ TEC+_Disk_6_ TEC+_Disk_7_ TEC+_Disk_8_ TEC+_Disk_9_ TEC-_Disk_1_ TEC-_Disk_2_ TEC-_Disk_3_ TEC-_Disk_4_ TEC-_Disk_5_ TEC-_Disk_6_ TEC-_Disk_7_ TEC-_Disk_8_ TEC-_Disk_9_ ;
#for partName0 in Tracker TIB TID TOB TEC ;

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
  if [ -f $outFileTK ]; then rm -f $outFileTK; fi
  touch $outFileTK
  echo Processing subDetName $subDetName, partType $partType, partNumber $partNumber
 
  for fileName in `ls $workdir/*/*/$Dataset/$baseName` ;
  do
#    echo $fileName
        # Extract run number from first row of file
        runNumber=`grep "New IOV" $fileName | awk '{print $6}'`
        line=`grep -A 10000 "Global Info" $fileName | grep -B 10000 Detid | grep "$partName" | awk -F ":" '{print $2}'`
#	echo $line
#	echo $runNumber
        nBadModulesTK=`echo $line | awk {'print $1'}`
        nAllBadFibersTK=`echo $line | awk '{print $2}'`
        nAllBadAPVsTK=`echo $line | awk '{print $3}'`
        let nBadAPVsFromFibersTK=$nAllBadFibersTK*2
        let nBadAPVsTK=$nAllBadAPVsTK-$nBadAPVsFromFibersTK
        nAllBadStripsTK=`echo $line | awk '{print $4}'`
        let nBadStripsFromAPVsTK=$nAllBadAPVsTK*128;
        let nBadStripsTK=$nAllBadStripsTK-$nBadStripsFromAPVsTK
        echo $runNumber $nBadModulesTK $nAllBadFibersTK $nAllBadAPVsTK $nBadStripsTK $nBadStripsFromAPVsTK $nAllBadStripsTK >> $outFileTK
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
    if [ ! -d $workdir/Trends ]; then mkdir $workdir/Trends; fi
    if [ ! -d $workdir/Trends/$Dataset ]; then mkdir $workdir/Trends/$Dataset ; fi
    if [ ! -d $workdir/Trends/$Dataset/PCLBadComponents ]; then mkdir $workdir/Trends/$Dataset/PCLBadComponents ; fi

    RepositoryDir=$workdir/Trends/$Dataset/PCLBadComponents

    mv $outRootFileTK $RepositoryDir/.
    cd $RepositoryDir
    makePlots $outRootFileTK $outRootPlotsTK

    if [ ! -d $RepositoryDir/TIB ]; then mkdir $RepositoryDir/TIB ; fi
    for i in {1..4}; do
	for Plot in `ls *.png | grep TIBLayer$i`; do
	    if [ ! -d $RepositoryDir/TIB/Layer$i ]; then mkdir $RepositoryDir/TIB/Layer$i ; fi
	    mv $Plot $RepositoryDir/TIB/Layer$i;
	done
    done
    
    if [ ! -d $RepositoryDir/TOB ]; then mkdir $RepositoryDir/TOB ; fi
    for i in {1..6}; do
	for Plot in `ls *.png | grep TOBLayer$i`; do
	    if [ ! -d $RepositoryDir/TOB/Layer$i ]; then mkdir $RepositoryDir/TOB/Layer$i ; fi
	    mv $Plot $RepositoryDir/TOB/Layer$i;
	done
    done
    
    if [ ! -d $RepositoryDir/TID ]; then mkdir $RepositoryDir/TID ; fi
    if [ ! -d $RepositoryDir/TID/Side1 ]; then mkdir $RepositoryDir/TID/Side1 ; fi
    if [ ! -d $RepositoryDir/TID/Side2 ]; then mkdir $RepositoryDir/TID/Side2 ; fi
    for i in {1..3}; do
	for Plot in `ls *.png | grep TID-Disk$i`; do
	    if [ ! -d $RepositoryDir/TID/Side1/Disk$i ]; then mkdir $RepositoryDir/TID/Side1/Disk$i ; fi
	    mv $Plot $RepositoryDir/TID/Side1/Disk$i;
	done
	for Plot in `ls *.png | grep TID+Disk$i`; do
	    if [ ! -d $RepositoryDir/TID/Side2/Disk$i ]; then mkdir $RepositoryDir/TID/Side2/Disk$i ; fi
	    mv $Plot $RepositoryDir/TID/Side2/Disk$i;
	done
    done
    
    if [ ! -d $RepositoryDir/TEC ]; then mkdir $RepositoryDir/TEC ; fi
    if [ ! -d $RepositoryDir/TEC/Side1 ]; then mkdir $RepositoryDir/TEC/Side1 ; fi
    if [ ! -d $RepositoryDir/TEC/Side2 ]; then mkdir $RepositoryDir/TEC/Side2 ; fi
    for i in {1..9}; do
	for Plot in `ls *.png | grep TEC-Disk$i`; do
	    if [ ! -d $RepositoryDir/TEC/Side1/Disk$i ]; then mkdir $RepositoryDir/TEC/Side1/Disk$i ; fi
	    mv $Plot $RepositoryDir/TEC/Side1/Disk$i;
	done
	for Plot in `ls *.png | grep TEC+Disk$i`; do
	    if [ ! -d $RepositoryDir/TEC/Side2/Disk$i ]; then mkdir $RepositoryDir/TEC/Side2/Disk$i ; fi
	    mv $Plot $RepositoryDir/TEC/Side2/Disk$i;
	done
    done
    
    for Plot in `ls *.png | grep TIB`; do
	mv $Plot $RepositoryDir/TIB;
    done
    
    for Plot in `ls *.png | grep TOB`; do
	mv $Plot $RepositoryDir/TOB;
    done
    
    for Plot in `ls *.png | grep TID-`; do
	mv $Plot $RepositoryDir/TID/Side1;
    done
    
    for Plot in `ls *.png | grep TID+`; do
	mv $Plot $RepositoryDir/TID/Side2;
    done
    
    for Plot in `ls *.png | grep TEC-`; do
	mv $Plot $RepositoryDir/TEC/Side1;
    done
    
    for Plot in `ls *.png | grep TEC+`; do
	mv $Plot $RepositoryDir/TEC/Side2;
    done
    
#    for Plot in `ls *.png | grep Tracker`; do
#	mv $Plot $RepositoryDir;
#    done
    
    cd $homedir
fi
