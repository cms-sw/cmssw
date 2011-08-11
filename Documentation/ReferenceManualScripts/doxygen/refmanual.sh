#!/bin/tcsh

	#INIT VARIABLES
        set CMSSW_xyz = $1
	set BUILDAREA = $PWD
        set SCRIPTS = $BUILDAREA/utils
	set Association = $SCRIPTS/Association-V6.txt

echo "####################################"

	echo "Init kerberos. Enter password:"
	kinit -l 60h

echo "####################################"

	# Setting environment
	echo "Setting environment"
		
	rm -Rf log_$CMSSW_xyz $CMSSW_xyz
	mkdir log_$CMSSW_xyz
	echo "####################################"
	echo "## Log directory: $PWD/log_$CMSSW_xyz"
	echo "####################################"

	echo "Init graphviz"
	source $CMS_PATH/sw/slc4_ia32_gcc345/external/graphviz/2.16.1-cms2/etc/profile.d/init.csh

	echo "Creating directory: $CMSSW_xyz"
	cmsrel $CMSSW_xyz
        cd $CMSSW_xyz
        cmsenv

	echo "Checkouting packages:"
        PackageManagement.pl --rel $CMSSW_xyz --ignorepack "SCRAMToolbox config" | tee $BUILDAREA/log_$CMSSW_xyz/checkout.log
	cd ..

echo "####################################"

	# Generate documentation
	echo "1. Generating doxygen: Started!"
	echo $PWD
	cd utils/doxygen
	
	# Updating configfile. Replacing @CMSSW_xyz@ to current CMSSW version
	sed -e 's/@CMSSW_xyz@/'$CMSSW_xyz'/g' cfgfile > cfgfile_$CMSSW_xyz

	./doxygen cfgfile_$CMSSW_xyz # | tee $BUILDAREA/log_$CMSSW_xyz/doxygen.log
	rm -f cfgfile_$CMSSW_xyz
	cd ../../
	echo "1. Generating doxygen: Done!"

echo "####################################"

	# Generate configfiles
	echo "2. Generating config files page: Started!"
	time python $SCRIPTS/configfiles/configfiles.py $CMSSW_xyz | tee $BUILDAREA/log_$CMSSW_xyz/configfiles.log
	echo "2. Generating config files page: Done!"

echo "####################################"

	# Generate links in cff, cfg, cfi files
	echo "3. Generating links (Linker) in config files: Started!"
	time python $SCRIPTS/linker/linker.py $CMSSW_xyz | tee $BUILDAREA/log_$CMSSW_xyz/linker.log
	echo "3. Generating links (Linker) in config files: Done!"

echo "####################################"

	# Generationg tree view
	echo "4. Generating tree view for index page: Started!"
	mkdir $CMSSW_xyz/doc/html/splittedTree
	time python $SCRIPTS/tree/tree_splitted.py $Association $CMSSW_xyz sim fastsim reco dqm daq gen core calib hlt geometry analysis visualization operations l1 db | tee $BUILDAREA/log_$CMSSW_xyz/tree.log
	echo "4. Generating tree view for index page: Done!"

	echo "4.1. Copying JQuery files: Started!"
        cp -R $SCRIPTS/tree/jquery/ $CMSSW_xyz/doc/html/splittedTree
	echo "4.1. Copying JQuery files: Done!"

echo "####################################"
	
	#Creating and copying index file
	echo "5. Updating index page: Started!"
	cd $SCRIPTS/indexPage/
	time python indexPage.py $CMSSW_xyz
	cd ../..
	echo "5. Updating index page: Done!"
	
	echo "5.1. Copying index pages: Started!"
	cp $SCRIPTS/indexPage/index.html $CMSSW_xyz/doc/html
	cp $SCRIPTS/indexPage/index.php $CMSSW_xyz/doc/html
	rm $SCRIPTS/indexPage/index.php
	echo "5.1. Copying index pages: Done!"
	
echo "####################################"

	# Splitting files: namespaces.html configfiles.html annotated.html(classes)
	echo "6. Splitting files:"
	echo "Follow process: tail -f $BUILDAREA/log_$CMSSW_xyz/splitter.log"
	echo "6.1. namespaces.html: Started!"
	time python $SCRIPTS/splitter/splitter.py $CMSSW_xyz /doc/html/namespaces.html namespaceList_ > $BUILDAREA/log_$CMSSW_xyz/splitter.log
	echo "6.1. namespaces.html: Done!"

	echo "6.2. configfiles.html: Started!"
	time python $SCRIPTS/splitter/splitter.py $CMSSW_xyz /doc/html/configfiles.html configfilesList_ >> $BUILDAREA/log_$CMSSW_xyz/splitter.log
	echo "6.2. configfiles.html: Done!"

	echo "6.3. annotated.html (ClassList): Started!"
	time python $SCRIPTS/splitter/splitter.py $CMSSW_xyz /doc/html/annotated.html annotatedList_ >> $BUILDAREA/log_$CMSSW_xyz/splitter.log
	echo "6.4. annotated.html (ClassList): Done!"

echo "####################################"
echo "7. Creating CMSSW.index file (all files in this release for doxygen redirect.php)"
      find $CMSSW_xyz/doc/html/ -name "*.html" ! \( -name "*dir_*" -o -name "*globals_*" -o -name "*namespacemembers_*" -o -name "*functions_*" \) -print | sort > $CMSSW_xyz/$CMSSW_xyz.index

cd $BUILDAREA/$CMSSW_xyz
rm -vrf bin cfipython config external include lib logs python share src test tmp objs
cd ..


