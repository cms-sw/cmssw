#!/bin/bash

eval `scramv1 ru -sh`

HOSTNAME=$(echo `/bin/hostname` | sed 's/\//\\\//g')
echo "The hostname is = $HOSTNAME"

TEST_PATH=$(echo "${PWD}" | sed 's/\//\\\//g')
echo "The current directory is = $PWD"

MWC_LIB1="${LOCALRT}/lib/slc4_ia32_gcc345/libDQMSiPixelHistoricInfoClient.so"
echo "Looking for the MonitorWebClient library... $MWC_LIB1"
if [ ! -f $MWC_LIB1 ]; then
    echo "Not Found! Will pick it up from the release area..."
    MWC_LIB1="/afs/cern.ch/cms/Releases/CMSSW/prerelease/${CMSSW_VERSION}/lib/slc3_ia32_gcc323/libDQMSiPixelHistoricInfoClient.so"
else 
    echo "Found!"
fi

MWC_LIB=$(echo "$MWC_LIB1" | sed 's/\//\\\//g')
echo $MWC_LIB1

if [ -e profile.xml ]; then
    rm profile.xml
fi 
if [ -e SiPixelHistoricInfoClient.xml ]; then
    rm SiPixelHistoricInfoClient.xml
fi
if [ -e startHistoricInfoClient ]; then
    rm startHistoricInfoClient
fi

sed -e "s/.portn/1972/g" -e "s/.host/${HOSTNAME}/g" -e "s/.pwd/${TEST_PATH}/g" -e "s/.libpath/${MWC_LIB}/g" .profile.xml > profile.xml
sed -e "s/.portn/1972/g" -e "s/.host/${HOSTNAME}/g" -e "s/.pwd/${TEST_PATH}/g" -e "s/.libpath/${MWC_LIB}/g" .SiPixelHistoricInfoClient.xml > SiPixelHistoricInfoClient.xml 
sed -e "s/.portn/1972/g" -e "s/.host/${HOSTNAME}/g" -e "s/.pwd/${TEST_PATH}/g" -e "s/.libpath/${MWC_LIB}/g" .startHistoricInfoClient > startHistoricInfoClient

chmod 751 profile.xml
chmod 751 SiPixelHistoricInfoClient.xml
chmod 751 startHistoricInfoClient



