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

cp .oO[Alignment/OfflineValidation]Oo./macros/FitPVResolution.C .
cp .oO[Alignment/OfflineValidation]Oo./macros/CMS_lumi.C .
cp .oO[Alignment/OfflineValidation]Oo./macros/CMS_lumi.h .

 if [[ .oO[pvresolutionreference]Oo. == *store* ]]; then xrdcp -f .oO[pvresolutionreference]Oo. PVValidation_reference.root; else ln -fs .oO[pvresolutionreference]Oo. ./PVResolution_reference.root; fi

root -b -q "FitPVResolution.C(\\"${PWD}/${RootOutputFile}=${theLabel},${PWD}/PVValidation_reference.root=Design simulation\\",\\"$theDate\\")"

mkdir -p .oO[plotsdir]Oo.
for PngOutputFile in $(ls *png ); do
    xrdcp -f ${PngOutputFile}  root://eoscms//eos/cms/store/caf/user/$USER/.oO[eosdir]Oo./plots/${PngOutputFile}
    rfcp ${PngOutputFile}  .oO[plotsdir]Oo.
done

for PdfOutputFile in $(ls *pdf ); do
    xrdcp -f ${PdfOutputFile}  root://eoscms//eos/cms/store/caf/user/$USER/.oO[eosdir]Oo./plots/${PdfOutputFile}
    rfcp ${PdfOutputFile}  .oO[plotsdir]Oo.
done

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

#include "Alignment/OfflineValidation/macros/FitPVResolution.C"

void TkAlPrimaryVertexResolutionPlot()
{

  // initialize the plot y-axis ranges
 .oO[PlottingInstantiation]Oo.
 FitPVResolution("","")

}
"""


