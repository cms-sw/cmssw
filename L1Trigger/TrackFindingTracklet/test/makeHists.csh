#!/bin/tcsh
###########################################################################################
# Create L1 track histograms & print summary of tracking performance.                     #
#                                                                                         #
# Run in folder where .root file (from L1TrackNtupleMaker_cfg.py) is located.             #
# The macro L1TrackNtuplePlot.C is assumed to be in this folder or the one above.         #
# If you want the histograms, you need to have a directory TrkPlots/ where they will go.  #
# If you only want the performance summary, this is not needed.                           #
#                                                                                         #
# Input root file name is arg.root if script run with argument "arg"                      #
#                      or TTbar_PU200_hybrid.root if no argument given.                   #
###########################################################################################

if ($#argv == 0) then
  set inputFileRoot = "TTbar_PU200_HYBRID"
else
  set inputFileRoot = $1
endif
set inputFileName = "${inputFileRoot}.root"
if ( -e $inputFileName) then
  echo "Processing $inputFileName"
else
  echo "ERROR: Input file $inputFileName not found"
  exit(1)
endif

# Find plotting macro
if ( -e L1TrackNtuplePlot.C ) then
  set DIR = '.'
else if ( -e ../L1TrackNtuplePlot.C ) then
  set DIR = '..'
else
  echo "ERROR: L1TrackNtuplePlot.C not found"
  exit(2)
endif

\root -b -q ${DIR}/'L1TrackNtuplePlot.C("'$inputFileRoot'")' | tail -n 18 >&! results.out 

cat results.out

echo "Tracking summary written to results.out"
echo "Histograms written to TrkPlots/"

exit
