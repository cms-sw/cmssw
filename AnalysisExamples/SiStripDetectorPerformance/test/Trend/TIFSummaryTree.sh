#!/bin/sh

TRACK=On
FLAG=Stable_FNAL_133_v7
base_path=/data1/CrabAnalysis/ClusterAnalysis

badList=bad.txt
#[ "$1" != "" ] && FLAG=$1
#[ "$2" != "" ] && outputFile=$2
[ "$1" != "" ] && TRACK=$1

outputFile=SummaryTree_${FLAG}_${TRACK}_good.root
outputRootFile=SummaryPlots_${FLAG}_${TRACK}_good.root

echo -e "&&&&&&&&&&&\nRunning TIFSummaryTree \n on flag $FLAG\n $TRACK writing file $outputFile \n&&&&&&&&&&&&&&&\n"

inputFileList=TIFSummaryTree_${FLAG}_good.list

rm -f $inputFileList
for file in `ls ${base_path}/$FLAG/ClusterAnalysis*/res/ClusterAnalysis*.root`
 do
runNb=`echo $file | awk -F"/" '{print $NF}' | sed -e "s@[a-Z]@@g" -e "s@_@@g"  -e "s@\.@@g"`
  [ "`grep -c $runNb $badList`" != "0" ] && continue
  echo -e "$runNb \t\t $file" >> $inputFileList
done

rm -f out_${FLAG}_${TRACK}

echo root -b -q -l "RunTIFSummaryTree_${TRACK}.C(\"$inputFileList\",\"$outputFile\")"
root -b -q -l "RunTIFSummaryTree_${TRACK}.C(\"$inputFileList\",\"$outputFile\")" > out_${FLAG}_${TRACK} 2>&1

echo root -b -q -l "RunPlotMacro_${TRACK}.C(\"$outputFile\", \"$outputRootFile\")"
root -b -q -l "RunPlotMacro_${TRACK}.C(\"$outputFile\", \"$outputRootFile\")" > Summary_${FLAG}_${TRACK}.txt


