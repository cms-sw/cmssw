#!/bin/sh

Tpath=/tmp/maborgia/ClusterThr
echo `ls $Tpath`
#for dir in `ls $Tpath`
#  do
#  [ ! -d $Tpath/$dir ] && continue

  for file in `ls $Tpath/$dir/*root`
    do

    outfile=`echo $file | sed -e "s@.root@_DoPlot.ps@g"`
    [ -e $outfile ] && continue
    echo -e "\n $file $outfile\n"
    root -b -l -q "DoPlot.C(\"$file\",\"$outfile\")"
  done
#done
