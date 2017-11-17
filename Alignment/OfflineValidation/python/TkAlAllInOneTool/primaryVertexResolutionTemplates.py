PrimaryVertexResolutionTemplate="""
## PV refit
process.load("TrackingTools.TransientTrack.TransientTrackBuilder_cfi")

from RecoVertex.PrimaryVertexProducer.OfflinePrimaryVertices_cfi import offlinePrimaryVertices 
process.offlinePrimaryVerticesFromRefittedTrks  = offlinePrimaryVertices.clone()
process.offlinePrimaryVerticesFromRefittedTrks.TrackLabel                                       = cms.InputTag("TrackRefitter") 
process.offlinePrimaryVerticesFromRefittedTrks.vertexCollections.maxDistanceToBeam              = 1
process.offlinePrimaryVerticesFromRefittedTrks.TkFilterParameters.maxNormalizedChi2             = 20
process.offlinePrimaryVerticesFromRefittedTrks.TkFilterParameters.minSiliconLayersWithHits      = 5
process.offlinePrimaryVerticesFromRefittedTrks.TkFilterParameters.maxD0Significance             = 5.0 
process.offlinePrimaryVerticesFromRefittedTrks.TkFilterParameters.minPixelLayersWithHits        = 2   

process.PrimaryVertexResolution = cms.EDAnalyzer('PrimaryVertexResolution',
                                                 vtxCollection       = cms.InputTag("offlinePrimaryVerticesFromRefittedTrks"),
                                                 trackCollection     = cms.InputTag("TrackRefitter"),		
                                                 minVertexNdf        = cms.untracked.double(10.),
                                                 minVertexMeanWeight = cms.untracked.double(0.5)
                                                 )

"""

####################################################################
####################################################################
PVResolutionPath="""
process.p = cms.Path(process.offlineBeamSpot                        + 
                     process.TrackRefitter                          + 
                     process.offlinePrimaryVerticesFromRefittedTrks +
                     process.PrimaryVertexResolution)
"""

####################################################################
####################################################################
PVResolutionScriptTemplate="""
#!/bin/bash
source /afs/cern.ch/cms/caf/setup.sh

echo  -----------------------
echo  Job started at `date`
echo  -----------------------

export theLabel=.oO[alignmentName]Oo.
export theDate = pippa

cwd=`pwd`
cd .oO[CMSSW_BASE]Oo./src
export SCRAM_ARCH=.oO[SCRAM_ARCH]Oo.
eval `scram runtime -sh`
cd $cwd

rfmkdir -p .oO[datadir]Oo.
rfmkdir -p .oO[workingdir]Oo.
rfmkdir -p .oO[logdir]Oo.
rm -f .oO[logdir]Oo./*.stdout
rm -f .oO[logdir]Oo./*.stderr

if [[ $HOSTNAME = lxplus[0-9]*[.a-z0-9]* ]] # check for interactive mode
then
    rfmkdir -p .oO[workdir]Oo.
    rm -f .oO[workdir]Oo./*
    cd .oO[workdir]Oo.
else
    mkdir -p $cwd/TkAllInOneTool
    cd $cwd/TkAllInOneTool
fi

.oO[CommandLine]Oo.

ls -lh .

eos mkdir -p /store/caf/user/$USER/.oO[eosdir]Oo./plots/
for RootOutputFile in $(ls *root )
do
    xrdcp -f ${RootOutputFile} root://eoscms//eos/cms/store/caf/user/$USER/.oO[eosdir]Oo./${RootOutputFile}
    rfcp ${RootOutputFile}  .oO[workingdir]Oo.
done

cp .oO[Alignment/OfflineValidation]Oo./macros/FitPVResiduals.C .
cp .oO[Alignment/OfflineValidation]Oo./macros/CMS_lumi.C .
cp .oO[Alignment/OfflineValidation]Oo./macros/CMS_lumi.h .

 if [[ .oO[pvresolutionreference]Oo. == *store* ]]; then xrdcp -f .oO[pvresolutionreference]Oo. PVValidation_reference.root; else ln -fs .oO[pvresolutionreference]Oo. ./PVResolution_reference.root; fi

root -b -q "FitPVResiduals.C(\\"${PWD}/${RootOutputFile}=${theLabel},${PWD}/PVValidation_reference.root=Design simulation\\",true,true,\\"$theDate\\")"

mkdir -p .oO[plotsdir]Oo.
for PngOutputFile in $(ls *png ); do
    xrdcp -f ${PngOutputFile}  root://eoscms//eos/cms/store/caf/user/$USER/.oO[eosdir]Oo./plots/${PngOutputFile}
    rfcp ${PngOutputFile}  .oO[plotsdir]Oo.
done

for PdfOutputFile in $(ls *pdf ); do
    xrdcp -f ${PdfOutputFile}  root://eoscms//eos/cms/store/caf/user/$USER/.oO[eosdir]Oo./plots/${PdfOutputFile}
    rfcp ${PdfOutputFile}  .oO[plotsdir]Oo.
done

mkdir .oO[plotsdir]Oo./Biases/
mkdir .oO[plotsdir]Oo./Biases/dzPhi
mkdir .oO[plotsdir]Oo./Biases/dxyPhi
mkdir .oO[plotsdir]Oo./Biases/dzEta
mkdir .oO[plotsdir]Oo./Biases/dxyEta
mkdir .oO[plotsdir]Oo./Fit
mkdir .oO[plotsdir]Oo./dxyVsEta
mkdir .oO[plotsdir]Oo./dzVsEta
mkdir .oO[plotsdir]Oo./dxyVsPhi
mkdir .oO[plotsdir]Oo./dzVsPhi
mkdir .oO[plotsdir]Oo./dxyVsEtaNorm
mkdir .oO[plotsdir]Oo./dzVsEtaNorm
mkdir .oO[plotsdir]Oo./dxyVsPhiNorm
mkdir .oO[plotsdir]Oo./dzVsPhiNorm

mv .oO[plotsdir]Oo./BiasesCanvas*     .oO[plotsdir]Oo./Biases/
mv .oO[plotsdir]Oo./dzPhiBiasCanvas*  .oO[plotsdir]Oo./Biases/dzPhi
mv .oO[plotsdir]Oo./dxyPhiBiasCanvas* .oO[plotsdir]Oo./Biases/dxyPhi
mv .oO[plotsdir]Oo./dzEtaBiasCanvas*  .oO[plotsdir]Oo./Biases/dzEta
mv .oO[plotsdir]Oo./dxyEtaBiasCanvas* .oO[plotsdir]Oo./Biases/dxyEta
mv .oO[plotsdir]Oo./dzPhiTrendFit*    .oO[plotsdir]Oo./Fit
mv .oO[plotsdir]Oo./dxyEtaTrendNorm*  .oO[plotsdir]Oo./dxyVsEtaNorm
mv .oO[plotsdir]Oo./dzEtaTrendNorm*   .oO[plotsdir]Oo./dzVsEtaNorm
mv .oO[plotsdir]Oo./dxyPhiTrendNorm*  .oO[plotsdir]Oo./dxyVsPhiNorm
mv .oO[plotsdir]Oo./dzPhiTrendNorm*   .oO[plotsdir]Oo./dzVsPhiNorm
mv .oO[plotsdir]Oo./dxyEtaTrend*      .oO[plotsdir]Oo./dxyVsEta
mv .oO[plotsdir]Oo./dzEtaTrend*       .oO[plotsdir]Oo./dzVsEta
mv .oO[plotsdir]Oo./dxyPhiTrend*      .oO[plotsdir]Oo./dxyVsPhi
mv .oO[plotsdir]Oo./dzPhiTrend*       .oO[plotsdir]Oo./dzVsPhi

wget https://raw.githubusercontent.com/mmusich/PVToolScripts/master/PolishedScripts/index.php

cp index.php .oO[plotsdir]Oo./Biases/
cp index.php .oO[plotsdir]Oo./Biases/dzPhi
cp index.php .oO[plotsdir]Oo./Biases/dxyPhi
cp index.php .oO[plotsdir]Oo./Biases/dzEta
cp index.php .oO[plotsdir]Oo./Biases/dxyEta
cp index.php .oO[plotsdir]Oo./Fit
cp index.php .oO[plotsdir]Oo./dxyVsEta
cp index.php .oO[plotsdir]Oo./dzVsEta
cp index.php .oO[plotsdir]Oo./dxyVsPhi
cp index.php .oO[plotsdir]Oo./dzVsPhi
cp index.php .oO[plotsdir]Oo./dxyVsEtaNorm
cp index.php .oO[plotsdir]Oo./dzVsEtaNorm
cp index.php .oO[plotsdir]Oo./dxyVsPhiNorm
cp index.php .oO[plotsdir]Oo./dzVsPhiNorm


echo  -----------------------
echo  Job ended at `date`
echo  -----------------------

"""

######################################################################
######################################################################

PVResolutionPlotExecution="""
#make primary vertex validation plots

rfcp .oO[plottingscriptpath]Oo. .
root -x -b -q .oO[plottingscriptname]Oo.++

for PdfOutputFile in $(ls *pdf ); do
    xrdcp -f ${PdfOutputFile}  root://eoscms//eos/cms/store/caf/user/$USER/.oO[eosdir]Oo./plots/${PdfOutputFile}
    rfcp ${PdfOutputFile}  .oO[datadir]Oo./.oO[PlotsDirName]Oo.
done

for PngOutputFile in $(ls *png ); do
    xrdcp -f ${PngOutputFile}  root://eoscms//eos/cms/store/caf/user/$USER/.oO[eosdir]Oo./plots/${PngOutputFile}
    rfcp ${PngOutputFile}  .oO[datadir]Oo./.oO[PlotsDirName]Oo.
done

"""

######################################################################
######################################################################

PVResolutionPlotTemplate="""
/****************************************
This can be run directly in root, or you
 can run ./TkAlMerge.sh in this directory
****************************************/

#include "Alignment/OfflineValidation/macros/FitPVResiduals.C"

void TkAlPrimaryVertexResolutionPlot()
{

  // initialize the plot y-axis ranges
 .oO[PlottingInstantiation]Oo.

}
"""


