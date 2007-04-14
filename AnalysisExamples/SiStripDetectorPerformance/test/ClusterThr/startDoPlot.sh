#!/bin/sh


for file in `ls /tmp/giordano/ClusterThr/*root`
    do

  echo -e "\n $file \n"

  outfile=`echo $file | sed -e "s@.root@_DoPlot.ps@g"`
  root -b -l -q "DoPlot.C(\"$file\",\"$outfile\")"
done
