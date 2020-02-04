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
# Deterministic annealing clustering
####################################################################
if isDA:
     print(">>>>>>>>>> testPVValidation_cfg.py: msg%-i: Running DA Algorithm!")
     process.PVValidation = cms.EDAnalyzer("PrimaryVertexValidation",
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
                                           
                                           TkFilterParameters = cms.PSet(algorithm=cms.string('filter'),                           
                                                                         maxNormalizedChi2 = cms.double(5.0),                        # chi2ndof < 5                  
                                                                         minPixelLayersWithHits = cms.int32(2),                      # PX hits > 2                       
                                                                         minSiliconLayersWithHits = cms.int32(5),                    # TK hits > 5  
                                                                         maxD0Significance = cms.double(5.0),                        # fake cut (requiring 1 PXB hit)     
                                                                         minPt = cms.double(0.0),                                    # better for softish events                        
                                                                         maxEta = cms.double(5.0),                                   # as per recommendation in PR #18330
                                                                         trackQuality = cms.string("any")
                                                                         ),

                                           ## MM 04.05.2017 (use settings as in: https://github.com/cms-sw/cmssw/pull/18330)
                                           TkClusParameters=cms.PSet(algorithm=cms.string('DA_vect'),
                                                                     TkDAClusParameters = cms.PSet(coolingFactor = cms.double(0.6),  # moderate annealing speed
                                                                                                   Tmin = cms.double(2.0),           # end of vertex splitting
                                                                                                   Tpurge = cms.double(2.0),         # cleaning
                                                                                                   Tstop = cms.double(0.5),          # end of annealing
                                                                                                   vertexSize = cms.double(0.006),   # added in quadrature to track-z resolutions
                                                                                                   d0CutOff = cms.double(3.),        # downweight high IP tracks
                                                                                                   dzCutOff = cms.double(3.),        # outlier rejection after freeze-out (T<Tmin)
                                                                                                   zmerge = cms.double(1e-2),        # merge intermediat clusters separated by less than zmerge
                                                                                                   uniquetrkweight = cms.double(0.8) # require at least two tracks with this weight at T=Tpurge
                                                                                                   )
                                                                     )
                                           )

####################################################################
# GAP clustering
####################################################################
else:
     print(">>>>>>>>>> testPVValidation_cfg.py: msg%-i: Running GAP Algorithm!")
     process.PVValidation = cms.EDAnalyzer("PrimaryVertexValidation",
                                           TrackCollectionTag = cms.InputTag("FinalTrackRefitter"),
                                           VertexCollectionTag = cms.InputTag(".oO[VertexCollection]Oo."),
                                           Debug = cms.bool(False),
                                           isLightNtuple = cms.bool(True),
                                           storeNtuple = cms.bool(False),
                                           useTracksFromRecoVtx = cms.bool(False),
                                           askFirstLayerHit = cms.bool(False),
                                           forceBeamSpot = cms.untracked.bool(.oO[forceBeamSpot]Oo.),
                                           probePt = cms.untracked.double(.oO[ptCut]Oo.),
                                           probeEta = cms.untracked.double(.oO[etaCut]Oo.),
                                           doBPix   = cms.untracked.bool(.oO[doBPix]Oo.),
                                           doFPix   = cms.untracked.bool(.oO[doFPix]Oo.),
                                           numberOfBins = cms.untracked.int32(.oO[numberOfBins]Oo.),
                                           runControl = cms.untracked.bool(.oO[runControl]Oo.),
                                           runControlNumber = cms.untracked.vuint32(int(.oO[runboundary]Oo.)),

                                           TkFilterParameters = cms.PSet(algorithm=cms.string('filter'),
                                                                         maxNormalizedChi2 = cms.double(5.0),                        # chi2ndof < 20
                                                                         minPixelLayersWithHits=cms.int32(2),                        # PX hits > 2
                                                                         minSiliconLayersWithHits = cms.int32(5),                    # TK hits > 5
                                                                         maxD0Significance = cms.double(5.0),                        # fake cut (requiring 1 PXB hit)
                                                                         minPt = cms.double(0.0),                                    # better for softish events
                                                                         maxEta = cms.double(5.0),                                   # as per recommendation in PR #18330
                                                                         trackQuality = cms.string("any")
                                                                         ),

                                           TkClusParameters = cms.PSet(algorithm   = cms.string('gap'),
                                                                       TkGapClusParameters = cms.PSet(zSeparation = cms.double(0.2)  # 0.2 cm max separation betw. clusters
                                                                                                      )
                                                                       )
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


