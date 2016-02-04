#!/bin/bash

eval `scramv1 ru -sh`

HOSTNAME=$(echo `/bin/hostname` | sed 's/\//\\\//g')
#HOSTNAME=localhost
echo "The hostname is = $HOSTNAME"


TEST_PATH=$(echo "${PWD}" | sed 's/\//\\\//g')
echo "The current directory is = $PWD"

#MWC_LIB1="${LOCALRT}/lib/${SCRAM_ARCH}/libDQML1TMonitorClient.so"
MWC_LIB1="${LOCALRT}/lib/${SCRAM_ARCH}/pluginDQML1TMonitorClient.so"
echo "Looking for the L1TMonitorClient library... $MWC_LIB1"
if [ ! -f $MWC_LIB1 ]; then
    echo "Not Found! Will pick it up from the release area..."
    MWC_LIB1="/afs/cern.ch/cms/Releases/CMSSW/prerelease/${CMSSW_VERSION}/lib/${SCRAM_ARCH}/libDQML1TMonitorClient.so"
else 
    echo "Found!"
fi

MWC_LIB=$(echo "$MWC_LIB1" | sed 's/\//\\\//g')
echo $MWC_LIB1

if [ -e profile.xml ]; then
    rm profile.xml
fi 

if [ -e L1TClient.xml ]; then
    rm L1TClient.xml
fi
if [ -e startMonitorClient ]; then
    rm startMonitorClient
fi

sed -e "s/.portn/1972/g" -e "s/.host/${HOSTNAME}/g" -e "s/.pwd/${TEST_PATH}/g" -e "s/.libpath/${MWC_LIB}/g" .profile.xml > profile.xml
sed -e "s/.portn/1972/g" -e "s/.host/${HOSTNAME}/g" -e "s/.pwd/${TEST_PATH}/g" -e "s/.libpath/${MWC_LIB}/g" .L1TClient.xml > L1TClient.xml 
sed -e "s/.portn/1972/g" -e "s/.host/${HOSTNAME}/g" -e "s/.pwd/${TEST_PATH}/g" -e "s/.libpath/${MWC_LIB}/g" .startMonitorClient > startMonitorClient

chmod 751 profile.xml
chmod 751 L1TClient.xml
chmod 751 startMonitorClient



