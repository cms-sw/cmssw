#!/bin/bash
#

# Script to prepare the tables in TkAlignmentPixelPosition.
#
# In the beginning you have to state
# - the paths to the template config for the geometry comparison and
#   the plotting macros
# - which run numbers to test
# - which global tags and possibly which tracker alignment tags in which DB paths

#CONFIG_TEMPLATE="src/Alignment/MillePedeAlignmentAlgorithm/test/alignment_forGeomComp_cfg_TEMPLATE.py"
#PLOTMILLEPEDEDIR=
CONFIG_TEMPLATE="$CMSSW_BASE/src/Alignment/TrackerAlignment/test/alignment_forGeomComp_cfg_TEMPLATE.py"
PLOTMILLEPEDEDIR="$CMSSW_BASE/src/Alignment/MillePedeAlignmentAlgorithm/macros/"
if [ ! -f "${CONFIG_TEMPLATE}" ]
then
    CONFIG_TEMPLATE="${CMSSW_RELEASE_BASE}/src/Alignment/TrackerAlignment/test/alignment_forGeomComp_cfg_TEMPLATE.py"
fi
if [ ! -d "${PLOTMILLEPEDEDIR}" ]
then
    PLOTMILLEPEDEDIR="${CMSSW_RELEASE_BASE}/src/Alignment/MillePedeAlignmentAlgorithm/macros/"
fi

if [ $# -gt 0 ]
then
    EXECUTION_DIR="${1}/"
else
    EXECUTION_DIR="$(pwd)/"
fi

echo Using template $CONFIG_TEMPLATE 
echo and plotting macros from $PLOTMILLEPEDEDIR
echo "Running in ${EXECUTION_DIR}"
echo 

RUN_NUMBERS="272011 273000"

# First conditions to check
# (if ALIGNMENT_TAG1 and DB_PATH_TAG1 are empty takes content from GLOBALTAG1)
# also symbolic Global Tags are allowed
GLOBALTAG1="auto:run2_data"
# GLOBALTAG1="90X_dataRun2_Express_v4"
ALIGNMENT_TAG1="TrackerAlignment_2009_v1_express"
DB_PATH_TAG1="frontier://FrontierProd/CMS_CONDITIONS"

# Second conditions to check
# also symbolic Global Tags are allowed
GLOBALTAG2="auto:run2_data"
# GLOBALTAG2="90X_dataRun2_Express_v4"
ALIGNMENT_TAG2="SiPixelAli_PCL_v0_prompt"
# ALIGNMENT_TAG2="SiPixelAli_PCL_v0_p"
DB_PATH_TAG2="frontier://FrontierPrep/CMS_CONDITIONS"

if [ ! -d "${EXECUTION_DIR}" ]
then
    mkdir ${EXECUTION_DIR}
fi
cd ${EXECUTION_DIR}

for RUN in $RUN_NUMBERS ; do
    echo "============================================================"
    echo " Run $RUN: $GLOBALTAG1 / $ALIGNMENT_TAG1 (=1) vs $GLOBALTAG2 / $ALIGNMENT_TAG2 (=2)" 
    echo "============================================================"
    CONFIG1=alignment_forGeomComp_${GLOBALTAG1}_${ALIGNMENT_TAG1}_r${RUN}_1.py
    TREEFILE1=treeFile_${GLOBALTAG1}_${ALIGNMENT_TAG1}_r${RUN}_1.root
    TREEFILE1=`echo ${TREEFILE1//"auto:"/"auto_"}`
    LOGFILE1=alignment_${GLOBALTAG1}_${ALIGNMENT_TAG1}r${RUN}_1
    LOGFILE1=`echo ${LOGFILE1//"auto:"/"auto_"}`
    #echo $TREEFILE1 $LOGFILE1

    CONFIG2=alignment_forGeomComp_${GLOBALTAG2}_${ALIGNMENT_TAG2}_r${RUN}_2.py
    TREEFILE2=treeFile_${GLOBALTAG2}_${ALIGNMENT_TAG2}_r${RUN}_2.root
    TREEFILE2=`echo ${TREEFILE2//"auto:"/"auto_"}`
    LOGFILE2=alignment_${GLOBALTAG2}_${ALIGNMENT_TAG2}r${RUN}_2
    LOGFILE2=`echo ${LOGFILE2//"auto:"/"auto_"}`
    #echo $TREEFILE2 $LOGFILE2

    if [ -e $TREEFILE1 ] ; then
	echo "Removing old file" $TREEFILE1
	rm $TREEFILE1
    fi
    sed -e "s/RUNNUMBER/${RUN}/g" $CONFIG_TEMPLATE > ${CONFIG1}_tmp 
    sed -e "s/TREEFILE/${TREEFILE1}/g" ${CONFIG1}_tmp > ${CONFIG1}_tmp2
    sed -e "s/GLOBALTAG/${GLOBALTAG1}/g" ${CONFIG1}_tmp2 > ${CONFIG1}_tmp3
    sed -e "s/LOGFILE/${LOGFILE1}/g" ${CONFIG1}_tmp3 > ${CONFIG1}
  
    # maybe we need to overwrite GlobalTag alignment?
    if [ "$ALIGNMENT_TAG1" != "" ]; then
	cat >> ${CONFIG1} <<EOF

from CondCore.CondDB.CondDB_cfi import *
CondDBReference = CondDB.clone(connect = cms.string("$DB_PATH_TAG1"))
process.trackerAlignment = cms.ESSource("PoolDBESSource",
                                        CondDBReference,
                                        toGet = cms.VPSet(cms.PSet(record = cms.string("TrackerAlignmentRcd"),
                                                                   tag = cms.string("$ALIGNMENT_TAG1")
                                                                  )
                                                         )       
                                       )
process.es_prefer_trackerAlignment = cms.ESPrefer("PoolDBESSource", "trackerAlignment")
EOF
    fi

    rm ${CONFIG1}_tmp* 
    cmsRun $CONFIG1
    return_code=${?}
    if [ ${return_code} -ne 0 ]
    then
	echo "The command 'cmsRun ${CONFIG1}' failed. Please check the log file."
	exit ${return_code}
    fi
    rm remove_me.db

    if [ -e $TREEFILE2 ] ; then
	echo "Removing old file" $TREEFILE2
	rm $TREEFILE2
    fi
    sed -e "s/RUNNUMBER/${RUN}/g" $CONFIG_TEMPLATE > ${CONFIG2}_tmp 
    sed -e "s/TREEFILE/${TREEFILE2}/g" ${CONFIG2}_tmp > ${CONFIG2}_tmp2
    sed -e "s/GLOBALTAG/${GLOBALTAG2}/g" ${CONFIG2}_tmp2 > ${CONFIG2}_tmp3
    sed -e "s/LOGFILE/${LOGFILE2}/g" ${CONFIG2}_tmp3 > ${CONFIG2}
   
    # maybe we need to overwrite GlobalTag alignment?
    if [ "$ALIGNMENT_TAG2" != "" ]; then
	cat >> ${CONFIG2} <<EOF

from CondCore.CondDB.CondDB_cfi import *
CondDBReference = CondDB.clone(connect = cms.string("$DB_PATH_TAG2"))
process.trackerAlignment = cms.ESSource("PoolDBESSource",
                                        CondDBReference,
                                        toGet = cms.VPSet(cms.PSet(record = cms.string("TrackerAlignmentRcd"),
                                                                   tag = cms.string("$ALIGNMENT_TAG2")
                                                                  )
                                                         )       
                                       )
process.es_prefer_trackerAlignment = cms.ESPrefer("PoolDBESSource", "trackerAlignment")
EOF
    fi

    rm ${CONFIG2}_tmp*
    cmsRun $CONFIG2
    return_code=${?}
    if [ ${return_code} -ne 0 ]
    then
	echo "The command 'cmsRun ${CONFIG2}' failed. Please check the log file."
	exit ${return_code}
    fi
    rm remove_me.db

    HEREIAM=$(pwd)
    PLOTDIR=${HEREIAM}/PixelBaryCentrePlottingTools
    if [ ! -d ${PLOTDIR} ]
    then
       mkdir ${PLOTDIR}
       cp -r $PLOTMILLEPEDEDIR/* ${PLOTDIR}
       chmod -R +w ${PLOTDIR}
    fi
    cd ${PLOTDIR}
    root -b -q -l allMillePede.C "pixelPositionChange.C+(\"${HEREIAM}/$TREEFILE1\", \"${HEREIAM}/$TREEFILE2\")"
    return_code=${?}
    if [ ${return_code} -ne 0 ]
    then
	echo "Running 'allMillePede.C' failed."
	exit ${return_code}
    fi
    cd $HEREIAM
done
