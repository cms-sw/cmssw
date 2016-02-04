#!/bin/bash

export TNS_ADMIN=/etc

OUT_LOG=/nfshome0/hcallumipro/LumiLog/DBS-Transfer/DBS-Transfer.log

echo "==================== " >> ${OUT_LOG}

date >> ${OUT_LOG}

echo "==================== " >> ${OUT_LOG}

echo "Directory $1" >> ${OUT_LOG}
echo "File name $2" >> ${OUT_LOG}

echo "=========DBS======= " >> ${OUT_LOG}

lumiFileXfer-v2.sh -d $1 -f $2 >> ${OUT_LOG} 2>&1
