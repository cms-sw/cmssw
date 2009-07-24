#!/bin/sh
#
# 07-Jan-2009, KAB - script to configure hosts within a SM development system.

# check for valid arguments
if [ "$1" == "-h" ]
then
    echo "Usage: configureHosts.sh [baseXdaqPort [hostNameOverride]]"
    echo "e.g. 'configureHosts.sh 50000 cmsroc8.fnal.gov'"
    exit
fi
if [ $# -gt 0 ]
then
    basePort=$1
else
    basePort=50000
fi
if [ $# -gt 1 ]
then
    localHostName=$2
else
    localHostName=$HOSTNAME
fi
buPort=$(($basePort + 0))
fu1Port=$(($basePort + 1))
fu2Port=$(($basePort + 2))
fu3Port=$(($basePort + 3))
fu4Port=$(($basePort + 4))
fu5Port=$(($basePort + 5))
fu6Port=$(($basePort + 6))
fu7Port=$(($basePort + 7))
fu8Port=$(($basePort + 8))
smPort=$(($basePort + 1000))
smpsPort=$(($basePort + 1001))
consPort=$(($basePort + 1002))
buTcpPort=$(($basePort + 1010))
smTcpPort=$(($basePort + 1011))

fuCfgFile="fu_twoOut.py"
smCfgFile="sm_streams.py"
consCfgFile="fuConsumer.py"

for filename in `find -maxdepth 3 -name "*.base"`
do
    finalFile="${filename/.base}"
    inputFile="$filename"

    sed -e "{
        s/STMGR_DEV_BU_HOST/$localHostName/;
        s/STMGR_DEV_BU_PORT/$buPort/;

        s/STMGR_DEV_FU_HOST/$localHostName/;
        s/STMGR_DEV_FU_PORT/$fu1Port/;

        s/STMGR_DEV_FU2_HOST/$localHostName/;
        s/STMGR_DEV_FU2_PORT/$fu2Port/;

        s/STMGR_DEV_FU3_HOST/$localHostName/;
        s/STMGR_DEV_FU3_PORT/$fu3Port/;

        s/STMGR_DEV_FU4_HOST/$localHostName/;
        s/STMGR_DEV_FU4_PORT/$fu4Port/;

        s/STMGR_DEV_FU5_HOST/$localHostName/;
        s/STMGR_DEV_FU5_PORT/$fu5Port/;

        s/STMGR_DEV_FU6_HOST/$localHostName/;
        s/STMGR_DEV_FU6_PORT/$fu6Port/;

        s/STMGR_DEV_FU7_HOST/$localHostName/;
        s/STMGR_DEV_FU7_PORT/$fu7Port/;

        s/STMGR_DEV_FU8_HOST/$localHostName/;
        s/STMGR_DEV_FU8_PORT/$fu8Port/;

        s/STMGR_DEV_SM_HOST/$localHostName/;
        s/STMGR_DEV_SM_PORT/$smPort/;

        s/STMGR_DEV_SMPROXY_HOST/$localHostName/;
        s/STMGR_DEV_SMPROXY_PORT/$smpsPort/;

        s/STMGR_DEV_CONSFU_HOST/$localHostName/;
        s/STMGR_DEV_CONSFU_PORT/$consPort/;

        s/BU_ENDPOINT_PORT/$buTcpPort/;
        s/SM_ENDPOINT_PORT/$smTcpPort/;

        s/FU_CFG_FILE/$fuCfgFile/;
        s/SM_CFG_FILE/$smCfgFile/;

        s/CONS_CFG_FILE/$consCfgFile/;

      }" $inputFile > $finalFile


    if [[ $finalFile =~ "(csh|sh|tcl)$" ]]
    then
        chmod +x $finalFile
    fi
done
