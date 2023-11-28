#!/bin/tcsh
########################################################################
# Create L1 track histograms & print summary of tracking performance,  #
# by running ROOT macros L1TrackNtuplePlot.C & L1TrackQualityPlot.C    #
# on the .root file from L1TrackNtupleMaker_cfg.py .                   # #                                                                      #
#                                                                      #
# To use:                                                              #
#   makeHists.csh  rootFileName                                        #
#                                                                      #
# (where rootFileName is the name of the input .root file,             #
#  including its directory name, if its not in the current one.        #
#  If rootFileName not specified, it defaults to TTbar_PU200_D76.root) #
########################################################################

if ($#argv == 0) then
  set inputFullFileName = "TTbar_PU200_D88.root"
else
  set inputFullFileName = $1
endif

if ( -e $inputFullFileName) then
  echo "Processing $inputFullFileName"
else
  echo "ERROR: Input file $inputFullFileName not found"
  exit(1)
endif

# Get directory name
set dirName = `dirname $inputFullFileName`/
# Get file name without directory name
set fileName = `basename $inputFullFileName`
# Get stem of filename, removing ".root".
set inputFileStem = `echo $fileName | awk -F . '{print $1;}'`

eval `scramv1 runtime -csh`

# Run track quality MVA plotting macro
set plotMacro = $CMSSW_BASE/src/L1Trigger/TrackFindingTracklet/test/L1TrackQualityPlot.C
if (-e MVA_plots) rm -r MVA_plots
\root -b -q ${plotMacro}'("'${inputFileStem}'","'${dirName}'")' 
echo "MVA track quality Histograms written to MVA_plots/"  

# Run track performance plotting macro
set plotMacro = $CMSSW_BASE/src/L1Trigger/TrackFindingTracklet/test/L1TrackNtuplePlot.C
if (-e TrkPlots) rm -r TrkPlots
\root -b -q ${plotMacro}'("'${inputFileStem}'","'${dirName}'")' | tail -n 19 >! results.out 
cat results.out
echo "Tracking performance summary written to results.out"
echo "Track performance histograms written to TrkPlots/"  

exit
