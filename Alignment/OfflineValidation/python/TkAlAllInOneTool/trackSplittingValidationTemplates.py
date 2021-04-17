######################################################################
######################################################################
TrackSplittingTemplate="""

#adding this ~doubles the efficiency of selection
process.FittingSmootherRKP5.EstimateCut = -1

.oO[subdetselection]Oo.

# Use compressions settings of TFile
# see https://root.cern.ch/root/html534/TFile.html#TFile:SetCompressionSettings
# settings = 100 * algorithm + level
# level is from 1 (small) to 9 (large compression)
# algo: 1 (ZLIB), 2 (LMZA)
# see more about compression & performance: https://root.cern.ch/root/html534/guides/users-guide/InputOutput.html#compression-and-performance
compressionSettings = 207
process.cosmicValidation = cms.EDAnalyzer("CosmicSplitterValidation",
    compressionSettings = cms.untracked.int32(compressionSettings),
    ifSplitMuons = cms.bool(False),
    ifTrackMCTruth = cms.bool(False),	
    checkIfGolden = cms.bool(False),	
    splitTracks = cms.InputTag("FinalTrackRefitter","","splitter"),
    splitGlobalMuons = cms.InputTag("muons","","splitter"),
    originalTracks = cms.InputTag("FirstTrackRefitter","","splitter"),
    originalGlobalMuons = cms.InputTag("muons","","Rec")
)
"""

######################################################################
######################################################################
TrackSplittingSequence = "process.cosmicValidation"


######################################################################
######################################################################
trackSplitPlotExecution="""
#make track splitting plots

cp .oO[trackSplitPlotScriptPath]Oo. .
root -x -b -q TkAlTrackSplitPlot.C++

"""

######################################################################
######################################################################

trackSplitPlotTemplate="""
#include "Alignment/OfflineValidation/macros/trackSplitPlot.C"

/****************************************
This can be run directly in root, or you
 can run ./TkAlMerge.sh in this directory
It can be run as is, or adjusted to fit
 for misalignments or to only make
 certain plots
****************************************/

/********************************
To make ALL plots (247 in total):
  leave this file as is
********************************/

/**************************************************************************
to make all plots involving a single x or y variable, or both:
Uncomment the line marked (B), and fill in for xvar and yvar

Examples:

   xvar = "dxy", yvar = "ptrel" - makes plots of dxy vs Delta_pT/pT
                                  (4 total - profile and resolution,
                                   of Delta_pT/pT and its pull
                                   distribution)
   xvar = "all",   yvar = "pt"  - makes all plots involving Delta_pT
                                  (not Delta_pT/pT)
                                  (30 plots total:
                                   histogram and pull distribution, and
                                   their mean and width as a function
                                   of the 7 x variables)
   xvar = "",      yvar = "all" - makes all histograms of all y variables
                                  (including Delta_pT/pT)
                                  (16 plots total - 8 y variables,
                                   regular and pull histograms)
**************************************************************************/

/**************************************************************************************
To make a custom selection of plots:
Uncomment the lines marked (C) and this section, and fill in matrix however you want */

/*
Bool_t plotmatrix[xsize][ysize];
void fillmatrix()
{
    for (int x = 0; x < xsize; x++)
        for (int y = 0; y < ysize; y++)
            plotmatrix[x][y] = (.............................);
}
*/

/*
The variables are defined in Alignment/OfflineValidation/macros/trackSplitPlot.h
 as follows:
TString xvariables[xsize]      = {"", "pt", "eta", "phi", "dz",  "dxy", "theta",
                                  "qoverpt"};

TString yvariables[ysize]      = {"pt", "pt",  "eta", "phi", "dz",  "dxy", "theta",
                                  "qoverpt", ""};
Bool_t relativearray[ysize]    = {true, false, false, false, false, false, false,
                                  false,     false};
Use matrix[x][y] = true to make that plot, and false not to make it.
**************************************************************************************/

/*************************************************************************************
To fit for a misalignment, which can be combined with any other option:
Uncomment the line marked (A) and this section, and choose your misalignment        */

/*
TString misalignment = "choose one";
double *values = 0;
double *phases = 0;
//or:
//    double values[number of files] = {...};
//    double phases[number of files] = {...};
*/

/*
The options for misalignment are sagitta, elliptical, skew, telescope, or layerRot.
If the magnitude and phase of the misalignment are known (i.e. Monte Carlo data using
 a geometry produced by the systematic misalignment tool), make values and phases into
 arrays, with one entry for each file, to make a plot of the result of the fit vs. the
 misalignment value.
phases must be filled in for sagitta, elliptical, and skew if values is;
 for the others it has no effect
*************************************************************************************/

void TkAlTrackSplitPlot()
{
    TkAlStyle::legendheader = ".oO[legendheader]Oo.";
    TkAlStyle::legendoptions = ".oO[legendoptions]Oo.";
    TkAlStyle::set(.oO[publicationstatus]Oo., .oO[era]Oo., ".oO[customtitle]Oo.", ".oO[customrighttitle]Oo.");
    outliercut = .oO[outliercut]Oo.;
    //fillmatrix();                                                         //(C)
    subdetector = ".oO[subdetector]Oo.";
    makePlots(
.oO[PlottingInstantiation]Oo.
              ,
              //misalignment,values,phases,                                 //(A)
              ".oO[datadir]Oo./.oO[PlotsDirName]Oo."
              //,"xvar","yvar"                                              //(B)
              //,plotmatrix                                                 //(C)
             );
}
"""
