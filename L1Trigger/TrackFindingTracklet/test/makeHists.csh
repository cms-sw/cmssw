#!/bin/tcsh
###########################################################################
# Create L1 track histograms & print summary of tracking performance,     #
# by running ROOT macro L1TrackNtuplePlot.C on .root file                 #
# from L1TrackNtupleMaker_cfg.py                                          #                     
#                                                                         #
# To use:                                                                 #
#   makeHists.csh  rootFileName                                           #
#                                                                         #
# (where rootFileName is the name of the input .root file,                #
#  including its directory name, if its not in the current one.           #
#  If rootFileName not specified, it defaults to TTbar_PU200_hybrid.root) #
###########################################################################

if ($#argv == 0) then
  set inputFullFileName = "TTbar_PU200_D49.root"
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

# Find plotting macro
eval `scramv1 runtime -csh`
set plotMacro = $CMSSW_BASE/src/L1Trigger/TrackFindingTracklet/test/L1TrackNtuplePlot.C
if ( -e $plotMacro ) then
  # Run plotting macro
  if (-e TrkPlots) rm -r TrkPlots
  \root -b -q ${plotMacro}'("'${inputFileStem}'","'${dirName}'")' | tail -n 19 >! results.out 
  cat results.out
  echo "Tracking performance summary written to results.out"
  echo "Histograms written to TrkPlots/"  
else if ( -e ../L1TrackNtuplePlot.C ) then
else
  echo "ERROR: $plotMacro not found"
  exit(2)
endif

exit
