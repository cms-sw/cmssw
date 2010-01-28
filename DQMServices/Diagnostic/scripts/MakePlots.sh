#!/bin/sh

ExeName=$1
Database=$2
TagName=$3
Password=$4
RunStart=$5
RunEnd=$6

PlotDir="CurrentPlots"

rm -rf $PlotDir

if [ $6 ]; then
    echo "RUNNING: $ExeName $Database $TagName $Password $RunStart $RunEnd"
    $ExeName $Database $TagName $Password $RunStart $RunEnd
else
    echo "RUNNING: $ExeName $Database $TagName $Password $RunStart"
    $ExeName $Database $TagName $Password $RunStart
fi
mkdir -pv $PlotDir
mv *.gif $PlotDir
mv *.eps $PlotDir
cp DeanConvert.pl $PlotDir
cp html/$ExeName.html $PlotDir/index.html
cd $PlotDir
./DeanConvert.pl
rm -f DeanConvert.pl

cd ..

