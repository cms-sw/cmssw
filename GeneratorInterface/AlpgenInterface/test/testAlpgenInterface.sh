#!/bin/bash

# This is going to be a script that performs a test suit over the 
# AlpgenInterface package. Still in development.
# Any questions, please contact Thiago Tomei (IFT-UNESP / SPRACE). 

# Modify these defaults to suit your environment.
ARCHITECTURE=slc4_ia32_gcc345
ALPGEN_PATH=$CMS_PATH/sw/$ARCHITECTURE/external/alpgen/213-cms
ALPGEN_BIN_PATH=$ALPGEN_PATH/bin

ls $ALPGEN_BIN_PATH
