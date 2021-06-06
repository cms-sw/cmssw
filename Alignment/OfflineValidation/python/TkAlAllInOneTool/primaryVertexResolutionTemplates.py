PrimaryVertexResolutionTemplate="""

HLTSel = .oO[doTriggerSelection]Oo.

###################################################################
#  Runs and events
###################################################################
runboundary = .oO[runboundary]Oo.
isMultipleRuns=False
if(isinstance(runboundary, (list, tuple))):
     isMultipleRuns=True
     print "Multiple Runs are selected"

if(isMultipleRuns):
     process.source.firstRun = cms.untracked.uint32(int(runboundary[0]))
else:
     process.source.firstRun = cms.untracked.uint32(int(runboundary))


###################################################################
# The trigger filter module
###################################################################
from HLTrigger.HLTfilters.triggerResultsFilter_cfi import *
process.theHLTFilter = triggerResultsFilter.clone(
    triggerConditions = cms.vstring(.oO[triggerBits]Oo.),
    hltResults = cms.InputTag( "TriggerResults", "", "HLT" ),
    l1tResults = cms.InputTag( "" ),
    throw = cms.bool(False)
)

###################################################################
# PV refit
###################################################################
process.load("TrackingTools.TransientTrack.TransientTrackBuilder_cfi")

from RecoVertex.PrimaryVertexProducer.OfflinePrimaryVertices_cfi import offlinePrimaryVertices 
process.offlinePrimaryVerticesFromRefittedTrks  = offlinePrimaryVertices.clone()
process.offlinePrimaryVerticesFromRefittedTrks.TrackLabel                                       = cms.InputTag("TrackRefitter") 
process.offlinePrimaryVerticesFromRefittedTrks.vertexCollections.maxDistanceToBeam              = 1
process.offlinePrimaryVerticesFromRefittedTrks.TkFilterParameters.maxNormalizedChi2             = 20
process.offlinePrimaryVerticesFromRefittedTrks.TkFilterParameters.minSiliconLayersWithHits      = 5
process.offlinePrimaryVerticesFromRefittedTrks.TkFilterParameters.maxD0Significance             = 5.0
# as it was prior to https://github.com/cms-sw/cmssw/commit/c8462ae4313b6be3bbce36e45373aa6e87253c59
process.offlinePrimaryVerticesFromRefittedTrks.TkFilterParameters.maxD0Error                    = 1.0
process.offlinePrimaryVerticesFromRefittedTrks.TkFilterParameters.maxDzError                    = 1.0
process.offlinePrimaryVerticesFromRefittedTrks.TkFilterParameters.minPixelLayersWithHits        = 2   

# Use compressions settings of TFile
# see https://root.cern.ch/root/html534/TFile.html#TFile:SetCompressionSettings
# settings = 100 * algorithm + level
# level is from 1 (small) to 9 (large compression)
# algo: 1 (ZLIB), 2 (LMZA)
# see more about compression & performance: https://root.cern.ch/root/html534/guides/users-guide/InputOutput.html#compression-and-performance
compressionSettings = 207

###################################################################
# The PV resolution module
###################################################################
process.PrimaryVertexResolution = cms.EDAnalyzer('SplitVertexResolution',
                                                 compressionSettings = cms.untracked.int32(compressionSettings),
                                                 storeNtuple         = cms.bool(False),
                                                 vtxCollection       = cms.InputTag("offlinePrimaryVerticesFromRefittedTrks"),
                                                 trackCollection     = cms.InputTag("TrackRefitter"),		
                                                 minVertexNdf        = cms.untracked.double(10.),
                                                 minVertexMeanWeight = cms.untracked.double(0.5),
                                                 runControl = cms.untracked.bool(.oO[runControl]Oo.),
                                                 runControlNumber = cms.untracked.vuint32(runboundary)
                                                 )

"""

####################################################################
####################################################################
PVResolutionPath="""

process.theValidSequence = cms.Sequence(process.offlineBeamSpot                        +
                                        process.TrackRefitter                          +
                                        process.offlinePrimaryVerticesFromRefittedTrks +
                                        process.PrimaryVertexResolution)
if (HLTSel):
    process.p = cms.Path(process.theHLTFilter + process.theValidSequence)
else:
    process.p = cms.Path(process.theValidSequence)
"""

####################################################################
####################################################################
PVResolutionScriptTemplate="""#!/bin/bash
source /afs/cern.ch/cms/caf/setup.sh
export X509_USER_PROXY=.oO[scriptsdir]Oo./.user_proxy

source /afs/cern.ch/cms/caf/setup.sh

echo  -----------------------
echo  Job started at `date`
echo  -----------------------

export theLabel=.oO[alignmentName]Oo.
export theDate=.oO[runboundary]Oo.

cwd=`pwd`
cd .oO[CMSSW_BASE]Oo./src
export SCRAM_ARCH=.oO[SCRAM_ARCH]Oo.
eval `scram runtime -sh`
cd $cwd

mkdir -p .oO[datadir]Oo.
mkdir -p .oO[workingdir]Oo.
mkdir -p .oO[logdir]Oo.
rm -f .oO[logdir]Oo./*.stdout
rm -f .oO[logdir]Oo./*.stderr

if [[ $HOSTNAME = lxplus[0-9]*[.a-z0-9]* ]] # check for interactive mode
then
    mkdir -p .oO[workdir]Oo.
    rm -f .oO[workdir]Oo./*
    cd .oO[workdir]Oo.
else
    mkdir -p $cwd/TkAllInOneTool
    cd $cwd/TkAllInOneTool
fi

.oO[CommandLine]Oo.

ls -lh .

eos mkdir -p /store/group/alca_trackeralign/AlignmentValidation/.oO[eosdir]Oo./plots/

for RootOutputFile in $(ls *root )
do
    xrdcp -f ${RootOutputFile} root://eoscms//eos/cms/store/group/alca_trackeralign/AlignmentValidation/.oO[eosdir]Oo./${RootOutputFile}
    cp ${RootOutputFile}  .oO[workingdir]Oo.
done

cp .oO[Alignment/OfflineValidation]Oo./macros/FitPVResolution.C .
cp .oO[Alignment/OfflineValidation]Oo./macros/CMS_lumi.C .
cp .oO[Alignment/OfflineValidation]Oo./macros/CMS_lumi.h .

 if [[ .oO[pvresolutionreference]Oo. == *store* ]]; then xrdcp -f .oO[pvresolutionreference]Oo. PVValidation_reference.root; else ln -fs .oO[pvresolutionreference]Oo. ./PVResolution_reference.root; fi

root -b -q "FitPVResolution.C(\\"${PWD}/${RootOutputFile}=${theLabel},${PWD}/PVValidation_reference.root=Design simulation\\",\\"$theDate\\")"

mkdir -p .oO[plotsdir]Oo.
for PngOutputFile in $(ls *png ); do
    xrdcp -f ${PngOutputFile}  root://eoscms//eos/cms/store/group/alca_trackeralign/AlignmentValidation/.oO[eosdir]Oo./plots/${PngOutputFile}
    cp ${PngOutputFile}  .oO[plotsdir]Oo.
done

for PdfOutputFile in $(ls *pdf ); do
    xrdcp -f ${PdfOutputFile}  root://eoscms//eos/cms/store/group/alca_trackeralign/AlignmentValidation/.oO[eosdir]Oo./plots/${PdfOutputFile}
    cp ${PdfOutputFile}  .oO[plotsdir]Oo.
done

echo  -----------------------
echo  Job ended at `date`
echo  -----------------------

"""

######################################################################
######################################################################

PVResolutionPlotExecution="""
#make primary vertex validation plots

cp .oO[plottingscriptpath]Oo. .
root -x -b -q .oO[plottingscriptname]Oo.++

for PdfOutputFile in $(ls *pdf ); do
    xrdcp -f ${PdfOutputFile}  root://eoscms//eos/cms/store/group/alca_trackeralign/AlignmentValidation/.oO[eosdir]Oo./plots/${PdfOutputFile}
    cp ${PdfOutputFile}  .oO[datadir]Oo./.oO[PlotsDirName]Oo.
done

for PngOutputFile in $(ls *png ); do
     xrdcp -f ${PngOutputFile}  root://eoscms//eos/cms/store/group/alca_trackeralign/AlignmentValidation/.oO[eosdir]Oo./plots/${PngOutputFile}
    cp ${PngOutputFile}  .oO[datadir]Oo./.oO[PlotsDirName]Oo.
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
 FitPVResolution("","");

}
"""


