PrimaryVertexValidationTemplate="""
import FWCore.ParameterSet.Config as cms
import sys
 
isDA = .oO[isda]Oo.
isMC = .oO[ismc]Oo.

process = cms.Process("PrimaryVertexValidation") 

###################################################################
# Event source and run selection
###################################################################
.oO[datasetDefinition]Oo.

###################################################################
#  Runs and events
###################################################################
runboundary = .oO[runboundary]Oo.
process.source.firstRun = cms.untracked.uint32(int(runboundary))

###################################################################
# JSON Filtering
###################################################################
if isMC:
     print ">>>>>>>>>> testPVValidation_cfg.py: msg%-i: This is simulation!"
     runboundary = 1
else:
     print ">>>>>>>>>> testPVValidation_cfg.py: msg%-i: This is real dATA!"
     if ('.oO[lumilist]Oo.'):
          print ">>>>>>>>>> testPVValidation_cfg.py: msg%-i: JSON filtering with: .oO[lumilist]Oo. "
          import FWCore.PythonUtilities.LumiList as LumiList
          process.source.lumisToProcess = LumiList.LumiList(filename ='.oO[lumilist]Oo.').getVLuminosityBlockRange()

###################################################################
# Messages
###################################################################
process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.destinations = ['cout', 'cerr']
process.MessageLogger.cerr.FwkReport.reportEvery = 1000

####################################################################
# Produce the Transient Track Record in the event
####################################################################
process.load("TrackingTools.TransientTrack.TransientTrackBuilder_cfi")

####################################################################
# Get the Magnetic Field
####################################################################
process.load("Configuration.StandardSequences..oO[magneticField]Oo._cff")

###################################################################
# Geometry load
###################################################################
process.load("Configuration.Geometry.GeometryRecoDB_cff")

####################################################################
# Get the BeamSpot
####################################################################
process.load("RecoVertex.BeamSpotProducer.BeamSpot_cff")

####################################################################
# Get the GlogalTag
####################################################################
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, '.oO[GlobalTag]Oo.', '')

.oO[condLoad]Oo.
     
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
process.filterOutLowPt.runControlNumber = [runboundary]
                                
if isMC:
     process.goodvertexSkim = cms.Sequence(process.noscraping + process.filterOutLowPt)
else:
     process.goodvertexSkim = cms.Sequence(process.primaryVertexFilter + process.noscraping + process.filterOutLowPt)

####################################################################
# Load and Configure Measurement Tracker Event 
# (would be needed in case NavigationSchool is set != from null
####################################################################
#process.load("RecoTracker.MeasurementDet.MeasurementTrackerEventProducer_cfi") 
#process.MeasurementTrackerEvent.pixelClusterProducer = '.oO[TrackCollection]Oo.'
#process.MeasurementTrackerEvent.stripClusterProducer = '.oO[TrackCollection]Oo.'
#process.MeasurementTrackerEvent.inactivePixelDetectorLabels = cms.VInputTag()
#process.MeasurementTrackerEvent.inactiveStripDetectorLabels = cms.VInputTag()

####################################################################
# Load and Configure TrackRefitter
####################################################################
process.load("RecoTracker.TrackProducer.TrackRefitters_cff")
import RecoTracker.TrackProducer.TrackRefitters_cff
process.TrackRefitter = RecoTracker.TrackProducer.TrackRefitter_cfi.TrackRefitter.clone()
process.TrackRefitter.src = ".oO[TrackCollection]Oo."
process.TrackRefitter.TrajectoryInEvent = True
process.TrackRefitter.NavigationSchool = ''
process.TrackRefitter.TTRHBuilder = "WithAngleAndTemplate"

####################################################################
# Output file
####################################################################
process.TFileService = cms.Service("TFileService",
                                   fileName=cms.string(".oO[outputFile]Oo.")
                                  )                                    

####################################################################
# Deterministic annealing clustering
####################################################################
if isDA:
     print ">>>>>>>>>> testPVValidation_cfg.py: msg%-i: Running DA Algorithm!"
     process.PVValidation = cms.EDAnalyzer("PrimaryVertexValidation",
                                           TrackCollectionTag = cms.InputTag("TrackRefitter"),
                                           VertexCollectionTag = cms.InputTag(".oO[VertexCollection]Oo."),  
                                           Debug = cms.bool(False),
                                           storeNtuple = cms.bool(False),
                                           useTracksFromRecoVtx = cms.bool(False),
                                           isLightNtuple = cms.bool(True),
                                           askFirstLayerHit = cms.bool(False),
                                           probePt = cms.untracked.double(.oO[ptCut]Oo.),
                                           numberOfBins = cms.untracked.int32(.oO[numberOfBins]Oo.),
                                           runControl = cms.untracked.bool(.oO[runControl]Oo.),
                                           runControlNumber = cms.untracked.vuint32(int(.oO[runboundary]Oo.)),
                                           
                                           TkFilterParameters = cms.PSet(algorithm=cms.string('filter'),                           
                                                                         maxNormalizedChi2 = cms.double(5.0),                        # chi2ndof < 5                  
                                                                         minPixelLayersWithHits = cms.int32(2),                      # PX hits > 2                       
                                                                         minSiliconLayersWithHits = cms.int32(5),                    # TK hits > 5  
                                                                         maxD0Significance = cms.double(5.0),                        # fake cut (requiring 1 PXB hit)     
                                                                         minPt = cms.double(0.0),                                    # better for softish events                        
                                                                         trackQuality = cms.string("any")
                                                                         ),
                                           
                                           TkClusParameters=cms.PSet(algorithm=cms.string('DA'),
                                                                     TkDAClusParameters = cms.PSet(coolingFactor = cms.double(0.8),  # moderate annealing speed
                                                                                                   Tmin = cms.double(4.),            # end of annealing
                                                                                                   vertexSize = cms.double(0.05),    # ~ resolution / sqrt(Tmin)
                                                                                                   d0CutOff = cms.double(3.),        # downweight high IP tracks
                                                                                                   dzCutOff = cms.double(4.)         # outlier rejection after freeze-out (T<Tmin)
                                                                                                   )
                                                                     )
                                           )

####################################################################
# GAP clustering
####################################################################
else:
     print ">>>>>>>>>> testPVValidation_cfg.py: msg%-i: Running GAP Algorithm!"
     process.PVValidation = cms.EDAnalyzer("PrimaryVertexValidation",
                                           TrackCollectionTag = cms.InputTag("TrackRefitter"),
                                           VertexCollectionTag = cms.InputTag(".oO[VertexCollection]Oo."), 
                                           Debug = cms.bool(False),
                                           isLightNtuple = cms.bool(True),
                                           storeNtuple = cms.bool(False),
                                           useTracksFromRecoVtx = cms.bool(False),
                                           askFirstLayerHit = cms.bool(False),
                                           probePt = cms.untracked.double(.oO[ptCut]Oo.),
                                           numberOfBins = cms.untracked.int32(.oO[numberOfBins]Oo.),
                                           runControl = cms.untracked.bool(.oO[runControl]Oo.),
                                           runControlNumber = cms.untracked.vuint32(int(.oO[runboundary]Oo.)),
                                           
                                           TkFilterParameters = cms.PSet(algorithm=cms.string('filter'),                             
                                                                         maxNormalizedChi2 = cms.double(5.0),                        # chi2ndof < 20                  
                                                                         minPixelLayersWithHits=cms.int32(2),                        # PX hits > 2                   
                                                                         minSiliconLayersWithHits = cms.int32(5),                    # TK hits > 5                   
                                                                         maxD0Significance = cms.double(5.0),                        # fake cut (requiring 1 PXB hit)
                                                                         minPt = cms.double(0.0),                                    # better for softish events     
                                                                         trackQuality = cms.string("any")
                                                                         ),
                                        
                                           TkClusParameters = cms.PSet(algorithm   = cms.string('gap'),
                                                                       TkGapClusParameters = cms.PSet(zSeparation = cms.double(0.2)  # 0.2 cm max separation betw. clusters
                                                                                                      ) 
                                                                       )
                                           )

####################################################################
# Path
####################################################################
process.p = cms.Path(process.goodvertexSkim*
                     process.offlineBeamSpot*
                     #process.MeasurementTrackerEvent*
                     process.TrackRefitter*
                     process.PVValidation)

"""

####################################################################
####################################################################
PVValidationScriptTemplate="""
#!/bin/bash 
source /afs/cern.ch/cms/caf/setup.sh
eos='/afs/cern.ch/project/eos/installation/cms/bin/eos.select'

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

rfmkdir -p .oO[datadir]Oo.
rfmkdir -p .oO[workingdir]Oo.
rfmkdir -p .oO[logdir]Oo.
rm -f .oO[logdir]Oo./*.stdout
rm -f .oO[logdir]Oo./*.stderr

if [[ $HOSTNAME = lxplus[0-9]*\.cern\.ch ]] # check for interactive mode
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

$eos mkdir -p /store/caf/user/$USER/.oO[eosdir]Oo./plots/
for RootOutputFile in $(ls *root )
do
    xrdcp -f ${RootOutputFile}  root://eoscms//eos/cms/store/caf/user/$USER/.oO[eosdir]Oo./
    rfcp ${RootOutputFile}  .oO[workingdir]Oo.
done

cp .oO[Alignment/OfflineValidation]Oo./macros/FitPVResiduals.C . 
cp .oO[Alignment/OfflineValidation]Oo./macros/CMS_lumi.C .
cp .oO[Alignment/OfflineValidation]Oo./macros/CMS_lumi.h .

 if [[ .oO[pvvalidationreference]Oo. == *store* ]]; then xrdcp -f .oO[pvvalidationreference]Oo. PVValidation_reference.root; else ln -fs .oO[pvvalidationreference]Oo. ./PVValidation_reference.root; fi
 
root -b -q "FitPVResiduals.C(\\"${PWD}/${RootOutputFile}=${theLabel},${PWD}/PVValidation_reference.root=Design simulation\\",true,true,\\"$theDate\\")"

mkdir -p .oO[plotsdir]Oo.
for PngOutputFile in $(ls *png ); do
    xrdcp -f ${PngOutputFile}  root://eoscms//eos/cms/store/caf/user/$USER/.oO[eosdir]Oo./plots/
    rfcp ${PngOutputFile}  .oO[plotsdir]Oo.
done

for PdfOutputFile in $(ls *pdf ); do                                                                                                                                            
    xrdcp -f ${PdfOutputFile}  root://eoscms//eos/cms/store/caf/user/$USER/.oO[eosdir]Oo./plots/                                                                                
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

PrimaryVertexPlotExecution="""
#make primary vertex validation plots

cp .oO[Alignment/OfflineValidation]Oo./macros/CMS_lumi.C .
cp .oO[Alignment/OfflineValidation]Oo./macros/CMS_lumi.h .
rfcp .oO[PrimaryVertexPlotScriptPath]Oo. .
root -x -b -q TkAlPrimaryVertexValidationPlot.C++

for PdfOutputFile in $(ls *pdf ); do                                                                                                                                  
    xrdcp -f ${PdfOutputFile}  root://eoscms//eos/cms/store/caf/user/$USER/.oO[eosdir]Oo./plots/                                                                         
    rfcp ${PdfOutputFile}  .oO[datadir]Oo.                                                                                                                               
done 

for PngOutputFile in $(ls *png ); do
    xrdcp -f ${PngOutputFile}  root://eoscms//eos/cms/store/caf/user/$USER/.oO[eosdir]Oo./plots/
    rfcp ${PngOutputFile}  .oO[datadir]Oo.
done

"""

######################################################################
######################################################################

PrimaryVertexPlotTemplate="""
/****************************************
This can be run directly in root, or you
 can run ./TkAlMerge.sh in this directory
It can be run as is, or adjusted to fit
 for misalignments or to only make
 certain plots
****************************************/

#include "Alignment/OfflineValidation/macros/FitPVResiduals.C"

void TkAlPrimaryVertexValidationPlot()
{
  FitPVResiduals(".oO[PrimaryVertexPlotInstantiation]Oo.",true,true,"");
}
"""
