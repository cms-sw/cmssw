#!/bin/tcsh

set CMSSW_xyz = $1	# CMSSW version
set SCRIPTS = $2	# full path to utils folder i.e. /data/refman/CMSSW_4_3_0/src/Documentation/ReferenceManualScripts/doxygen/utils
set LOCALTOP = $3	# full path to buildarea i.e. /data/refman/CMSSW_4_3_0/
set ASSOCIATION = $SCRIPTS/Association-V6.txt

# Generate configfiles
time python $SCRIPTS/configfiles/configfiles.py $LOCALTOP

# Generate links in cff, cfg, cfi files
time python $SCRIPTS/linker/linker.py $LOCALTOP

#Generationg tree view
mkdir $LOCALTOP/doc/html/splittedTree
cp -R $SCRIPTS/tree/jquery/ $LOCALTOP/doc/html/splittedTree

# Generating tree views and index page
time python $SCRIPTS/indexpage/Association.py $CMSSW_xyz $LOCALTOP $SCRIPTS

cp $SCRIPTS/other/ReferenceManual.html $LOCALTOP/doc/html

# Splitting
time python $SCRIPTS/splitter/splitter.py $LOCALTOP /doc/html/namespaces.html namespaceList_ 
time python $SCRIPTS/splitter/splitter.py $LOCALTOP /doc/html/configfiles.html configfilesList_ 
time python $SCRIPTS/splitter/splitter.py $LOCALTOP /doc/html/annotated.html annotatedList_ 
