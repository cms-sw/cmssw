#!/bin/tcsh

#set CMSSW_xyz = $SCRAM_PROJECTVERSION
#set SCRIPTS = $LOCALTOP/src/Documentation/ReferenceManualScripts/doxygen/utils

set CMSSW_xyz = $1
set SCRIPTS = $PWD/src/Documentation/ReferenceManualScripts/doxygen/utils

set CURRENT=$PWD;
cd $SCRIPTS/doxygen

cp -t $CURRENT cfgfile footer.html header.html doxygen.css DoxygenLayout.xml doxygen ../../script_launcher.sh

cd $CURRENT

chmod +rwx doxygen
./doxygen cfgfile

chmod +rwx script_launcher.sh
./script_launcher.sh $CMSSW_xyz $SCRIPTS $PWD

rm cfgfile footer.html header.html doxygen.css DoxygenLayout.xml doxygen script_launcher.sh

echo "Reference Manual is generated."


