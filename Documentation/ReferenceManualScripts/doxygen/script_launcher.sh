#!/bin/tcsh

set CMSSW_xyz = $1	# CMSSW version
set SCRIPTS = $2	# full path to utils folder i.e. /data/refman/CMSSW_4_3_0/src/Documentation/ReferenceManualScripts/doxygen/utils
set LOCALTOP = $3	# full path to buildarea i.e. /data/refman/CMSSW_4_3_0/

# CLEANUP
if (-e cfgfile.conf ) then
	rm -f cfgfile.conf
endif

if (-e $LOCALTOP/doc/html/splittedTree ) then
	rm -Rf $LOCALTOP/doc/html/splittedTree
endif

if (-e $LOCALTOP/$CMSSW_xyz.index) then
	rm -f $LOCALTOP/$CMSSW_xyz.index
endif

if (-e $LOCALTOP/doc/html/ReferenceManual.html) then
	rm $LOCALTOP/doc/html/ReferenceManual.html
endif

# END CLEANUP

# Generate configfiles
time python $SCRIPTS/configfiles/configfiles.py $LOCALTOP

# Generate links in cff, cfg, cfi files
time python $SCRIPTS/linker/linker.py $LOCALTOP

#Generationg tree view

mkdir $LOCALTOP/doc/html/splittedTree
cp -R $SCRIPTS/jquery/ $LOCALTOP/doc/html/splittedTree

# Generating tree views and index page
time python $SCRIPTS/indexPage/Association.py $CMSSW_xyz $LOCALTOP $SCRIPTS

cp $SCRIPTS/other/ReferenceManual.html $LOCALTOP/doc/html

# Splitting
time python $SCRIPTS/splitter/splitter.py $LOCALTOP /doc/html/namespaces.html namespaceList_ 
time python $SCRIPTS/splitter/splitter.py $LOCALTOP /doc/html/configfiles.html configfilesList_ 
time python $SCRIPTS/splitter/splitter.py $LOCALTOP /doc/html/annotated.html annotatedList_ 

find $LOCALTOP/doc/html/ -name "*.html" ! \( -name "*dir_*" -o -name "*globals_*" -o -name "*namespacemembers_*" -o -name "*functions_*" \) -print | sort > $LOCALTOP/$CMSSW_xyz.index
