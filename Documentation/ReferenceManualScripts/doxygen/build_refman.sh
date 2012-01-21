#!/bin/tcsh

	set LOCALTOP = $PWD
	set CMSSW_xyz = `echo $LOCALTOP | awk -F "/" '{print $NF}'`
	set SCRIPTS = $LOCALTOP/src/Documentation/ReferenceManualScripts/doxygen/utils
	
	# CLEANUP (if needed)

	if (-e $SCRIPTS/doxygen/cfgfile.conf) then
		rm $SCRIPTS/doxygen/cfgfile.conf
	endif

	if (-e $LOCALTOP/doc/html/splittedTree ) then
		rm -Rf $LOCALTOP/doc/html/splittedTree
	endif

	if (-e $LOCALTOP/doc ) then
		rm -Rf $LOCALTOP/doc
		mkdir doc
	endif

	if (-e $LOCALTOP/$CMSSW_xyz.index) then
		rm -f $LOCALTOP/$CMSSW_xyz.index
	endif

	if (-e $LOCALTOP/doc/html/ReferenceManual.html) then
		rm $LOCALTOP/doc/html/ReferenceManual.html
	endif

	# END CLEANUP

	cd $SCRIPTS/doxygen

	sed -e 's|@CMSSW_BASE@|'$LOCALTOP'|g' cfgfile > cfgfile.conf
	chmod +rwx doxygen
	./doxygen cfgfile.conf
	
	cd $LOCALTOP
	
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
	time python $SCRIPTS/splitter/packageDocSplitter.py pages.html $LOCALTOP 

	find $LOCALTOP/doc/html/ -name "*.html" ! \( -name "*dir_*" -o -name "*globals_*" -o -name "*namespacemembers_*" -o -name "*functions_*" \) -print | sort > $LOCALTOP/$CMSSW_xyz.index


	

	
