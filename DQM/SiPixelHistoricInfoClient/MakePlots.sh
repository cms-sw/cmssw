#!/bin/sh

First=$1

PlotDir="CurrentPixelPlots"

if [ $2 ]; then
SiPixelHDQMInspector $First $2
else
SiPixelHDQMInspector $First
fi
mkdir -pv $PlotDir
mv *.gif $PlotDir
mv *.eps $PlotDir
cp diow.pl $PlotDir
cd $PlotDir
./diow.pl
cd ..

