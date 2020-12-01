PrimaryVertexValidationTemplate="""

isDA = .oO[isda]Oo.
isMC = .oO[ismc]Oo.

###################################################################
#  Runs and events
###################################################################
runboundary = .oO[runboundary]Oo.
isMultipleRuns=False
if(isinstance(runboundary, (list, tuple))):
     isMultipleRuns=True
     print("Multiple Runs are selected")
       
if(isMultipleRuns):
     process.source.firstRun = cms.untracked.uint32(int(runboundary[0]))
else:
     process.source.firstRun = cms.untracked.uint32(int(runboundary)) 

###################################################################
# JSON Filtering
###################################################################
if isMC:
     print(">>>>>>>>>> testPVValidation_cfg.py: msg%-i: This is simulation!")
     runboundary = 1
else:
     print(">>>>>>>>>> testPVValidation_cfg.py: msg%-i: This is real DATA!")
     if ('.oO[lumilist]Oo.'):
          print(">>>>>>>>>> testPVValidation_cfg.py: msg%-i: JSON filtering with: .oO[lumilist]Oo. ")
          import FWCore.PythonUtilities.LumiList as LumiList
          process.source.lumisToProcess = LumiList.LumiList(filename ='.oO[lumilist]Oo.').getVLuminosityBlockRange()

####################################################################
# Produce the Transient Track Record in the event
####################################################################
process.load("TrackingTools.TransientTrack.TransientTrackBuilder_cfi")

####################################################################
# Load and Configure event selection
####################################################################
process.primaryVertexFilter = cms.EDFilter("VertexSelector",
                                           src = cms.InputTag(".oO[VertexCollection]Oo."),
                                           cut = cms.string("!isFake && ndof > 4 && abs(z) <= 24 && position.Rho <= 2"),
                                           filter = cms.bool(True)
                                           )

process.noscraping = cms.EDFilter("FilterOutScraping",
                                  applyfilter = cms.untracked.bool(True),
                                  src =  cms.untracked.InputTag(".oO[TrackCollection]Oo."),
                                  debugOn = cms.untracked.bool(False),
                                  numtrack = cms.untracked.uint32(10),
                                  thresh = cms.untracked.double(0.25)
                                  )


process.load("Alignment.CommonAlignment.filterOutLowPt_cfi")
process.filterOutLowPt.src = ".oO[TrackCollection]Oo."
process.filterOutLowPt.ptmin = .oO[ptCut]Oo.
process.filterOutLowPt.runControl = .oO[runControl]Oo.
if(isMultipleRuns):
     process.filterOutLowPt.runControlNumber.extend((runboundary))
else:
     process.filterOutLowPt.runControlNumber = [runboundary]
                                
if isMC:
     process.goodvertexSkim = cms.Sequence(process.noscraping + process.filterOutLowPt)
else:
     process.goodvertexSkim = cms.Sequence(process.primaryVertexFilter + process.noscraping + process.filterOutLowPt)

####################################################################
# Imports of parameters
####################################################################
from RecoVertex.PrimaryVertexProducer.OfflinePrimaryVertices_cfi import offlinePrimaryVertices
## modify the parameters which differ
FilteringParams = offlinePrimaryVertices.TkFilterParameters.clone(
     maxNormalizedChi2 = 5.0,  # chi2ndof < 5
     maxD0Significance = 5.0,  # fake cut (requiring 1 PXB hit)
     maxEta = 5.0,             # as per recommendation in PR #18330
)

## MM 04.05.2017 (use settings as in: https://github.com/cms-sw/cmssw/pull/18330)
from RecoVertex.PrimaryVertexProducer.TkClusParameters_cff import DA_vectParameters
DAClusterizationParams = DA_vectParameters.clone()

GapClusterizationParams = cms.PSet(algorithm   = cms.string('gap'),
                                   TkGapClusParameters = cms.PSet(zSeparation = cms.double(0.2))  # 0.2 cm max separation betw. clusters
                                   )

####################################################################
# Deterministic annealing clustering or Gap clustering
####################################################################
def switchClusterizerParameters(da):
     if da:
          print(">>>>>>>>>> testPVValidation_cfg.py: msg%-i: Running DA Algorithm!")
          return DAClusterizationParams
     else:
          print(">>>>>>>>>> testPVValidation_cfg.py: msg%-i: Running GAP Algorithm!")
          return GapClusterizationParams

# Use compressions settings of TFile
# see https://root.cern.ch/root/html534/TFile.html#TFile:SetCompressionSettings
# settings = 100 * algorithm + level
# level is from 1 (small) to 9 (large compression)
# algo: 1 (ZLIB), 2 (LMZA)
# see more about compression & performance: https://root.cern.ch/root/html534/guides/users-guide/InputOutput.html#compression-and-performance
compressionSettings = 207

####################################################################
# Configure the PVValidation Analyzer module
####################################################################
process.PVValidation = cms.EDAnalyzer("PrimaryVertexValidation",
                                      compressionSettings = cms.untracked.int32(compressionSettings),
                                      TrackCollectionTag = cms.InputTag("FinalTrackRefitter"),
                                      VertexCollectionTag = cms.InputTag(".oO[VertexCollection]Oo."),
                                      Debug = cms.bool(False),
                                      storeNtuple = cms.bool(False),
                                      useTracksFromRecoVtx = cms.bool(False),
                                      isLightNtuple = cms.bool(True),
                                      askFirstLayerHit = cms.bool(False),
                                      forceBeamSpot = cms.untracked.bool(.oO[forceBeamSpot]Oo.),
                                      probePt  = cms.untracked.double(.oO[ptCut]Oo.),
                                      probeEta = cms.untracked.double(.oO[etaCut]Oo.),
                                      doBPix   = cms.untracked.bool(.oO[doBPix]Oo.),
                                      doFPix   = cms.untracked.bool(.oO[doFPix]Oo.),
                                      numberOfBins = cms.untracked.int32(.oO[numberOfBins]Oo.),
                                      runControl = cms.untracked.bool(.oO[runControl]Oo.),
                                      runControlNumber = cms.untracked.vuint32(runboundary),
                                      TkFilterParameters = FilteringParams,
                                      TkClusParameters = switchClusterizerParameters(isDA)
                                      )
"""

####################################################################
####################################################################
PVValidationPath="""
process.p = cms.Path(process.goodvertexSkim*
                     process.seqTrackselRefit*
                     process.PVValidation)
"""

####################################################################
####################################################################
PVValidationScriptTemplate="""#!/bin/bash
source /afs/cern.ch/cms/caf/setup.sh
export X509_USER_PROXY=.oO[scriptsdir]Oo./.user_proxy

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

cp .oO[Alignment/OfflineValidation]Oo./macros/FitPVResiduals.C .
cp .oO[Alignment/OfflineValidation]Oo./macros/CMS_lumi.C .
cp .oO[Alignment/OfflineValidation]Oo./macros/CMS_lumi.h .

 if [[ .oO[pvvalidationreference]Oo. == *store* ]]; then xrdcp -f .oO[pvvalidationreference]Oo. PVValidation_reference.root; else ln -fs .oO[pvvalidationreference]Oo. ./PVValidation_reference.root; fi

echo "I am going to produce the comparison with IDEAL geometry of ${RootOutputFile}"
root -b -q "FitPVResiduals.C++g(\\"${PWD}/${RootOutputFile}=${theLabel},${PWD}/PVValidation_reference.root=Design simulation\\",true,true,\\"$theDate\\")"

mkdir -p .oO[plotsdir]Oo.
for PngOutputFile in $(ls *png ); do
    xrdcp -f ${PngOutputFile}  root://eoscms//eos/cms/store/group/alca_trackeralign/AlignmentValidation/.oO[eosdir]Oo./plots/${PngOutputFile}
    cp ${PngOutputFile}  .oO[plotsdir]Oo.
done

for PdfOutputFile in $(ls *pdf ); do
    xrdcp -f ${PdfOutputFile}  root://eoscms//eos/cms/store/group/alca_trackeralign/AlignmentValidation/.oO[eosdir]Oo./plots/${PdfOutputFile}
    cp ${PdfOutputFile}  .oO[plotsdir]Oo.
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

PrimaryVertexPlotExecution="""
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

PrimaryVertexPlotTemplate="""
/****************************************
This can be run directly in root, or you
 can run ./TkAlMerge.sh in this directory
****************************************/

#include "Alignment/OfflineValidation/macros/FitPVResiduals.C"

void TkAlPrimaryVertexValidationPlot()
{

  // initialize the plot y-axis ranges
  thePlotLimits->init(.oO[m_dxyPhiMax]Oo.,         // mean of dxy vs Phi
                      .oO[m_dzPhiMax]Oo.,          // mean of dz  vs Phi
                      .oO[m_dxyEtaMax]Oo.,         // mean of dxy vs Eta
                      .oO[m_dzEtaMax]Oo.,          // mean of dz  vs Eta
                      .oO[m_dxyPhiNormMax]Oo.,     // mean of dxy vs Phi (norm)
                      .oO[m_dzPhiNormMax]Oo.,      // mean of dz  vs Phi (norm)
                      .oO[m_dxyEtaNormMax]Oo.,     // mean of dxy vs Eta (norm)
                      .oO[m_dzEtaNormMax]Oo.,      // mean of dz  vs Eta (norm)
                      .oO[w_dxyPhiMax]Oo.,         // width of dxy vs Phi
                      .oO[w_dzPhiMax]Oo.,          // width of dz  vs Phi
                      .oO[w_dxyEtaMax]Oo.,         // width of dxy vs Eta
                      .oO[w_dzEtaMax]Oo.,          // width of dz  vs Eta
                      .oO[w_dxyPhiNormMax]Oo.,     // width of dxy vs Phi (norm)
                      .oO[w_dzPhiNormMax]Oo.,      // width of dz  vs Phi (norm)
                      .oO[w_dxyEtaNormMax]Oo.,     // width of dxy vs Eta (norm)
                      .oO[w_dzEtaNormMax]Oo.       // width of dz  vs Eta (norm)
		      );

 .oO[PlottingInstantiation]Oo.
  FitPVResiduals("",.oO[stdResiduals]Oo.,.oO[doMaps]Oo.,"",.oO[autoLimits]Oo.);
}
"""


