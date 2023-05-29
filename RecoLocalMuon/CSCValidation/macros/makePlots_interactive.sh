#!/bin/bash 
##
## Call makePlots.C to make plots from root file output of  CSCValidation
## CSC DPG - Tim Cox - 12.07.2022
## Run it like
## ./makePlots.sh path_to_root_file
## The plots are png files and produced in the current directory
## Remains in root, so need '.q' to quit
##
ARG1=$1
echo "Called with arg = "${ARG1}
root 'makePlots.C( "'${ARG1}'" )'

