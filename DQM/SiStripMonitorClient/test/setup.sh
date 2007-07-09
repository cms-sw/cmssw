#!/bin/bash


eval `scramv1 ru -sh`

COLLECTOR_NODE=$1
echo $COLLECTOR_NODE

HOSTNAME=$(echo `/bin/hostname -f` | sed 's/\//\\\//g')
echo "The hostname is = $HOSTNAME"

TEST_PATH=$(echo "${PWD}" | sed 's/\//\\\//g')
echo "The current directory is = $PWD"

LIB_NAME="pluginDQMSiStripMonitorClient.so"
if [ $# -gt 1 ]; then
 LIB_NAME="libDQMSiStripMonitorClient.so"
fi

MWC_LIB1="${LOCALRT}/lib/$SCRAM_ARCH/$LIB_NAME"
echo "Looking for the MonitorWebClient library... $MWC_LIB1"
if [ ! -f $MWC_LIB1 ]; then
    echo "Not Found! Will pick it up from the release area..."
    MWC_LIB1="${CMSSW_RELEASE_BASE}/lib/slc3_ia32_gcc323/libDQMSiStripMonitorClient.so"
else 
    echo "Found!"
fi

MWC_LIB=$(echo "$MWC_LIB1" | sed 's/\//\\\//g')
echo $MWC_LIB1

SERVED_DIR="http://${HOSTNAME}:1972/temporary"

if [ -e profile.xml ]; then
    rm profile.xml
fi 
if [ -e monClient.xml ]; then
    rm ClientWithWebInterface.xml
fi
if [ -e startMonitorClient ]; then
    rm startMonitorClient
fi

sed -e "s/.portn/1972/g" -e "s/.host/${HOSTNAME}/g" -e "s/.pwd/${TEST_PATH}/g" -e "s/.libpath/${MWC_LIB}/g" .profile.xml > profile.xml
sed -e "s/.portn/1972/g" -e "s/.host/${HOSTNAME}/g" -e "s/.pwd/${TEST_PATH}/g" -e "s/.libpath/${MWC_LIB}/g"  -e "s/.collector/${COLLECTOR_NODE}/g" .SiStripClient.xml > SiStripClient.xml 
sed -e "s/.portn/1972/g" -e "s/.host/${HOSTNAME}/g" -e "s/.pwd/${TEST_PATH}/g" -e "s/.libpath/${MWC_LIB}/g" .startMonitorClient > startMonitorClient

sed -e "s@SERVED_DIRECTORY_URL@${SERVED_DIR}@g" .WebLib.js > WebLib.js
sed -e "s@.host@${HOSTNAME}@g" .trackermap.txt > trackermap.txt

chmod 751 profile.xml
chmod 751 SiStripClient.xml
chmod 751 startMonitorClient



