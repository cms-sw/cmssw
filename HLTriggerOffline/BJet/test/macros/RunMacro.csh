#!/bin/tcsh

#Check to see if the CMS environment is set up
if ($?CMSSW_BASE != 1) then
    echo "CMS environment not set up"
    exit
endif

set topdir=221v300p6

#Check if base directory already exists
if (-d ${topdir}) then
    echo "Directory already exists"
    exit
endif

mkdir ${topdir}
mkdir ${topdir}/hlt_BTagIP_Jet180
mkdir ${topdir}/hlt_BTagIP_DoubleJet120
mkdir ${topdir}/hlt_BTagIP_TripleJet70
mkdir ${topdir}/hlt_BTagIP_QuadJet40
mkdir ${topdir}/hlt_BTagIP_HT470
mkdir ${topdir}/hlt_BTagIP_Jet120_Relaxed
mkdir ${topdir}/hlt_BTagIP_DoubleJet60_Relaxed
mkdir ${topdir}/hlt_BTagIP_TripleJet40_Relaxed
mkdir ${topdir}/hlt_BTagIP_QuadJet30_Relaxed
mkdir ${topdir}/hlt_BTagIP_HT320_Relaxed
mkdir ${topdir}/hlt_BTagMu_Jet20_Calib
mkdir ${topdir}/hlt_BTagMu_DoubleJet120
mkdir ${topdir}/hlt_BTagMu_TripleJet70
mkdir ${topdir}/hlt_BTagMu_QuadJet40
mkdir ${topdir}/hlt_BTagMu_HT370
mkdir ${topdir}/hlt_BTagMu_DoubleJet60_Relaxed
mkdir ${topdir}/hlt_BTagMu_TripleJet40_Relaxed
mkdir ${topdir}/hlt_BTagMu_QuadJet30_Relaxed
mkdir ${topdir}/hlt_BTagMu_HT250_Relaxed

root -l -q PlotMacro.C

mv *hlt_BTagIP_Jet180*gif              ${topdir}/hlt_BTagIP_Jet180
mv *hlt_BTagIP_DoubleJet120*gif        ${topdir}/hlt_BTagIP_DoubleJet120
mv *hlt_BTagIP_TripleJet70*gif         ${topdir}/hlt_BTagIP_TripleJet70
mv *hlt_BTagIP_QuadJet40*gif           ${topdir}/hlt_BTagIP_QuadJet40
mv *hlt_BTagIP_HT470*gif               ${topdir}/hlt_BTagIP_HT470
mv *hlt_BTagIP_Jet120_Relaxed*gif      ${topdir}/hlt_BTagIP_Jet120_Relaxed
mv *hlt_BTagIP_DoubleJet60_Relaxed*gif ${topdir}/hlt_BTagIP_DoubleJet60_Relaxed
mv *hlt_BTagIP_TripleJet40_Relaxed*gif ${topdir}/hlt_BTagIP_TripleJet40_Relaxed
mv *hlt_BTagIP_QuadJet30_Relaxed*gif   ${topdir}/hlt_BTagIP_QuadJet30_Relaxed
mv *hlt_BTagIP_HT320_Relaxed*gif       ${topdir}/hlt_BTagIP_HT320_Relaxed
mv *hlt_BTagMu_Jet20_Calib*gif         ${topdir}/hlt_BTagMu_Jet20_Calib
mv *hlt_BTagMu_DoubleJet120*gif        ${topdir}/hlt_BTagMu_DoubleJet120
mv *hlt_BTagMu_TripleJet70*gif         ${topdir}/hlt_BTagMu_TripleJet70
mv *hlt_BTagMu_QuadJet40*gif           ${topdir}/hlt_BTagMu_QuadJet40
mv *hlt_BTagMu_HT370*gif               ${topdir}/hlt_BTagMu_HT370
mv *hlt_BTagMu_DoubleJet60_Relaxed*gif ${topdir}/hlt_BTagMu_DoubleJet60_Relaxed
mv *hlt_BTagMu_TripleJet40_Relaxed*gif ${topdir}/hlt_BTagMu_TripleJet40_Relaxed
mv *hlt_BTagMu_QuadJet30_Relaxed*gif   ${topdir}/hlt_BTagMu_QuadJet30_Relaxed
mv *hlt_BTagMu_HT250_Relaxed*gif       ${topdir}/hlt_BTagMu_HT250_Relaxed

cp ../HTML_Files/TopLevel.html                       ${topdir}/index.html
cp ../HTML_Files/hlt_BTagIP_Jet180.html              ${topdir}/hlt_BTagIP_Jet180/index.html
cp ../HTML_Files/hlt_BTagIP_DoubleJet120.html        ${topdir}/hlt_BTagIP_DoubleJet120/index.html
cp ../HTML_Files/hlt_BTagIP_TripleJet70.html         ${topdir}/hlt_BTagIP_TripleJet70/index.html
cp ../HTML_Files/hlt_BTagIP_QuadJet40.html           ${topdir}/hlt_BTagIP_QuadJet40/index.html
cp ../HTML_Files/hlt_BTagIP_HT470.html               ${topdir}/hlt_BTagIP_HT470/index.html
cp ../HTML_Files/hlt_BTagIP_Jet120_Relaxed.html      ${topdir}/hlt_BTagIP_Jet120_Relaxed/index.html
cp ../HTML_Files/hlt_BTagIP_DoubleJet60_Relaxed.html ${topdir}/hlt_BTagIP_DoubleJet60_Relaxed/index.html
cp ../HTML_Files/hlt_BTagIP_TripleJet40_Relaxed.html ${topdir}/hlt_BTagIP_TripleJet40_Relaxed/index.html
cp ../HTML_Files/hlt_BTagIP_QuadJet30_Relaxed.html   ${topdir}/hlt_BTagIP_QuadJet30_Relaxed/index.html
cp ../HTML_Files/hlt_BTagIP_HT320_Relaxed.html       ${topdir}/hlt_BTagIP_HT320_Relaxed/index.html
cp ../HTML_Files/hlt_BTagMu_Jet20_Calib.html         ${topdir}/hlt_BTagMu_Jet20_Calib/index.html
cp ../HTML_Files/hlt_BTagMu_DoubleJet120.html        ${topdir}/hlt_BTagMu_DoubleJet120/index.html
cp ../HTML_Files/hlt_BTagMu_TripleJet70.html         ${topdir}/hlt_BTagMu_TripleJet70/index.html
cp ../HTML_Files/hlt_BTagMu_QuadJet40.html           ${topdir}/hlt_BTagMu_QuadJet40/index.html
cp ../HTML_Files/hlt_BTagMu_HT370.html               ${topdir}/hlt_BTagMu_HT370/index.html
cp ../HTML_Files/hlt_BTagMu_DoubleJet60_Relaxed.html ${topdir}/hlt_BTagMu_DoubleJet60_Relaxed/index.html
cp ../HTML_Files/hlt_BTagMu_TripleJet40_Relaxed.html ${topdir}/hlt_BTagMu_TripleJet40_Relaxed/index.html
cp ../HTML_Files/hlt_BTagMu_QuadJet30_Relaxed.html   ${topdir}/hlt_BTagMu_QuadJet30_Relaxed/index.html
cp ../HTML_Files/hlt_BTagMu_HT250_Relaxed.html       ${topdir}/hlt_BTagMu_HT250_Relaxed/index.html
