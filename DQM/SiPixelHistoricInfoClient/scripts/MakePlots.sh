#!/bin/sh

TagName=$1
First=$2

PlotDir="CurrentPlots"

rm -rf $PlotDir

if [ $3 ]; then
SiPixelHDQMInspector $TagName $First $3
else
SiPixelHDQMInspector $TagName $First
fi
mkdir -pv $PlotDir
mv *.gif $PlotDir
mv *.eps $PlotDir
cp diow.pl $PlotDir
cd $PlotDir
./diow.pl
cd ..

