#!/bin/sh

First=2

PlotDir="CurrentPixelPlots"

SiPixelHDQMInspector 2
mkdir -pv $PlotDir
mv *.gif $PlotDir
mv *.eps $PlotDir
cp diow.pl $PlotDir
cd $PlotDir
./diow.pl
cd ..

