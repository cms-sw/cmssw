#!/bin/zsh 
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
echo Using template $CONFIG_TEMPLATE 
echo and plotting macros from $PLOTMILLEPEDEDIR
echo 

RUN_NUMBERS=(273000)

# First conditions to check
# (if ALIGNMENT_TAG1 and DB_PATH_TAG1 are empty takes content from GLOBALTAG1)
# also symbolic Global Tags are allowed
GLOBALTAG1="auto:run2_data"
#GLOBALTAG1="80X_dataRun2_Prompt_v8"
ALIGNMENT_TAG1="TrackerAlignment_2009_v1_express"
DB_PATH_TAG1="frontier://FrontierProd/CMS_CONDITIONS"

# Second conditions to check
# also symbolic Global Tags are allowed
GLOBALTAG2="auto:run2_data"
#GLOBALTAG2="80X_dataRun2_Prompt_v8"
ALIGNMENT_TAG2="SiPixelAli_PCL_v0_prompt"
DB_PATH_TAG2="frontier://FrontierPrep/CMS_CONDITIONS"

for RUN in $RUN_NUMBERS ; do
    echo "============================================================"
    echo " Run $RUN: $GLOBALTAG1 / $ALIGNMENT_TAG1 (=1) vs $GLOBALTAG2 / $ALIGNMENT_TAG2 (=2)" 
    echo "============================================================"
    CONFIG1=alignment_forGeomComp_${GLOBALTAG1}_${ALIGNMENT_TAG1}_r$RUN.py
    TREEFILE1=treeFile_${GLOBALTAG1}_${ALIGNMENT_TAG1}_r${RUN}.root
    TREEFILE1=`echo ${TREEFILE1//"auto:"/"auto_"}`
    LOGFILE1=alignment_${GLOBALTAG1}_${ALIGNMENT_TAG1}r${RUN}
    LOGFILE1=`echo ${LOGFILE1//"auto:"/"auto_"}`
    #echo $TREEFILE1 $LOGFILE1

    CONFIG2=alignment_forGeomComp_${GLOBALTAG2}_${ALIGNMENT_TAG2}_r$RUN.py
    TREEFILE2=treeFile_${GLOBALTAG2}_${ALIGNMENT_TAG2}_r${RUN}.root
    TREEFILE2=`echo ${TREEFILE2//"auto:"/"auto_"}`
    LOGFILE2=alignment_${GLOBALTAG2}_${ALIGNMENT_TAG2}r${RUN}
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
    if [ $#ALIGNMENT_TAG1 != 0 ]; then
	cat >>! ${CONFIG1} <<EOF

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
    rm remove_me.db

    if [ -e $TREEFILE2 ] ; then
	echo "Removing old file " $TREEFILE2
	rm $TREEFILE2
    fi
    sed -e "s/RUNNUMBER/${RUN}/g" $CONFIG_TEMPLATE > ${CONFIG2}_tmp 
    sed -e "s/TREEFILE/${TREEFILE2}/g" ${CONFIG2}_tmp > ${CONFIG2}_tmp2
    sed -e "s/GLOBALTAG/${GLOBALTAG2}/g" ${CONFIG2}_tmp2 > ${CONFIG2}_tmp3
    sed -e "s/LOGFILE/${LOGFILE2}/g" ${CONFIG2}_tmp3 > ${CONFIG2}
   
    # maybe we need to overwrite GlobalTag alignment?
    if [ $#ALIGNMENT_TAG2 != 0 ]; then
	cat >>! ${CONFIG2} <<EOF

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
    rm remove_me.db

    HEREIAM=$(pwd)
    cd $PLOTMILLEPEDEDIR
    root -b -q -l allMillePede.C "pixelPositionChange.C+(\"${HEREIAM}/$TREEFILE1\", \"${HEREIAM}/$TREEFILE2\")"
    cd $HEREIAM
done
