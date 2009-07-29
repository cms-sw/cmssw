#!/bin/sh

TagName=$1
First=$2

PlotDir="CurrentPlots"

rm -rf $PlotDir

if [ $3 ]; then
SiStripHDQMInspector $TagName $First $3
else
SiStripHDQMInspector $TagName $First
fi
mkdir -pv $PlotDir
mv *.gif $PlotDir
mv *.eps $PlotDir
cp diow.pl $PlotDir
cd $PlotDir
./diow.pl
cd ..

