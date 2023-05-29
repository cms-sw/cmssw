#!/bin/bash 
##
## Call makePlots.C to make plots from root file output of CSCValidation
## CSC DPG - Tim Cox - 15.07.2022
## Run it like
## ./makePlots.sh path_to_root_file
## The plots are png files and produced in the current directory
##
ARG1=$1
echo "Called with arg = "${ARG1}

# run root in batch (-b), no banner (-l), quit after processing (-q)
root -b -l -q 'makePlots.C( "'${ARG1}'" )'
