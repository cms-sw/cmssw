#!/bin/sh

echo "Quick run and copy script by Freya.Blekman@cern.ch. Usage: \"run_and_to_castor.sh filelist.txt cmssconfigfile.cfg\""
echo "All CMSSW settings need to already have been done"
FILES=$1
CONFIGFILE=$2
#examine input file

CONFIGFILEINPUTNAME=`grep .dmp $CONFIGFILE | awk -F"file:" '{print $2}' | awk -F"\"" '{print $1}'`
CONFIGFILEOUTPUT=`grep .root $CONFIGFILE | awk -F"\"" '{print $2}'`
#echo "modifying output file "$CONFIGFILEOUTPUT" in "$CONFIGFILE
#echo "modifying input file "$CONFIGFILEINPUTNAME" in "$CONFIGFILE


#echo $FILES
for afile in `less $FILES | grep .dmp`
do
#    echo " "
    echo "now looking at file: " $afile
    pathname=${afile%/*}
    filesubname=${afile##*/}
    filenamenopostfix=`echo $filesubname | awk -F".dmp" '{print $1}'`
    filenamedigi=$pathname"/"$filenamenopostfix"_Digi.root"
    filenameraw=$pathname"/"$filenamenopostfix"_Raw.root"
    configfilename=$filenamenopostfix"_dmptoraw.cfg"
    cp -f $CONFIGFILE $configfilename
    replace $CONFIGFILEINPUTNAME $afile $CONFIGFILEOUTPUT $filenamedigi -- $configfilename
    echo $filenamedigi" is ready for running, type \"cmsRun "$filenamedigi"\" to start..."
done
