#!/bin/bash
#
# test record<->XML traslators, then use popcon to write sqlite file
# then read it back and write XML again. Finally test that initial and
# final XML are the same
#
# author: Stefano Argiro'
# $Id: testEcalConditionsPopCon.sh,v 1.1 2010/04/16 08:29:51 depasse Exp $
#


# test target directory and filenames are hardcoded, sigh !

echo Testing Ecal XML PopCon Conditions

eval `scramv1 runtime -sh`

#clean up
rm -f /tmp/Ecal*.xml
mkdir -p /tmp/sub/
rm -f /tmp/sub/Ecal*.xml
rm -f *.db

#test XML translator and write XML files
$CMSSW_BASE/test/slc4_ia32_gcc345/testXMLTranslators


#use PopCon to write sqlite files
for cfg in `ls testEcal*.py ` ; do
    cmsRun  $cfg
done

#read back sqlite files and write XML again
cmsRun testReaddbWriteXML.py

exstatus=0

#test that initial and final XML are the same
for file in `ls /tmp/Ecal*.xml` ; do
   diff --brief $file /tmp/sub/`basename $file`
   let exstatus+=$? 
done

if [ $exstatus -ne 0 ] ; then
  echo Test FAILED
else
  echo Test PASSED
fi

# clean up junk
rm -f /tmp/Ecal*.xml
rm -f /tmp/sub/Ecal*.xml
rm -f *.db
