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
CONFIG_TEMPLATE="$CMSSW_BASE/src/Alignment/MillePedeAlignmentAlgorithm/test/alignment_forGeomComp_cfg_TEMPLATE.py"
PLOTMILLEPEDEDIR="$CMSSW_BASE/src/Alignment/MillePedeAlignmentAlgorithm/macros/"
echo Using template $CONFIG_TEMPLATE 
echo and plotting macros from $PLOTMILLEPEDEDIR
echo 

# 2012A+B
#used in alignment ???:
#RUN_NUMBERS=(185189 190450 190702 190782 191718 191800 193093 194896 196197)
# in upload
#RUN_NUMBERS=(185189 190450 190702 190782 191691 191800 193093 194896 196197)
# 2012C
##RUN_NUMBERS=(197770 198230 198249 198346 200041 200229 200368 200532 201159 201191 201611 202012 202074 202972)
# 2012D
#RUN_NUMBERS=(203768 205086 205614 206187 207320 207779 208300)

RUN_NUMBERS=(248642)

# First conditions to check
# (if ALIGNMENT_TAG1 and DB_PATH_TAG1 are empty takes content from GLOBALTAG1)
GLOBALTAG1="GR_P_V56" #"FT_R_53_V6C"
ALIGNMENT_TAG1="TrackerAlignment_2009_v1_express"
DB_PATH_TAG1="frontier://FrontierProd/CMS_CONDITIONS"

# Second conditions to check
GLOBALTAG2="GR_P_V56" #"GR_R_53_V16D"
ALIGNMENT_TAG2="Alignments"
DB_PATH_TAG2="sqlite_file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERALIGN/PayLoads/2015-06-22_ForHighIntensity50nsRun/alignments_iter20.db"

for RUN in $RUN_NUMBERS ; do
    echo "============================================================"
    echo " Run $RUN: $GLOBALTAG1 / $ALIGNMENT_TAG1 (=1) vs $GLOBALTAG2 / $ALIGNMENT_TAG2 (=2)" 
    echo "============================================================"
    CONFIG1=alignment_forGeomComp_${GLOBALTAG1}_${ALIGNMENT_TAG1}_r$RUN.py
    TREEFILE1=treeFile_${GLOBALTAG1}_${ALIGNMENT_TAG1}_r${RUN}.root
    CONFIG2=alignment_forGeomComp_${GLOBALTAG2}_${ALIGNMENT_TAG2}_r$RUN.py
    TREEFILE2=treeFile_${GLOBALTAG2}_${ALIGNMENT_TAG2}_r${RUN}.root

    if [ -e $TREEFILE1 ] ; then
	echo "Removing old file" $TREEFILE1
	rm $TREEFILE1
    fi
    sed -e "s/RUNNUMBER/${RUN}/g" $CONFIG_TEMPLATE > ${CONFIG1}_tmp 
    sed -e "s/TREEFILE/${TREEFILE1}/g" ${CONFIG1}_tmp > ${CONFIG1}_tmp2
    sed -e "s/GLOBALTAG/${GLOBALTAG1}/g" ${CONFIG1}_tmp2 > ${CONFIG1}_tmp3
    sed -e "s/LOGFILE/alignment_${GLOBALTAG1}_${ALIGNMENT_TAG1}r${RUN}/g" ${CONFIG1}_tmp3 > ${CONFIG1}
    # maybe we need to overwrite GlobalTag alignment?
    if [ $#ALIGNMENT_TAG1 != 0 ]; then
	cat >>! ${CONFIG1} <<EOF

from CondCore.DBCommon.CondDBSetup_cfi import *
process.trackerAlignment = cms.ESSource(
    "PoolDBESSource",
    CondDBSetup,
    connect = cms.string("$DB_PATH_TAG1"),
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
    sed -e "s/LOGFILE/alignment_${GLOBALTAG2}_${ALIGNMENT_TAG2}r${RUN}/g" ${CONFIG2}_tmp3 > ${CONFIG2}

    # maybe we need to overwrite GlobalTag alignment?
    if [ $#ALIGNMENT_TAG2 != 0 ]; then
	cat >>! ${CONFIG2} <<EOF

from CondCore.DBCommon.CondDBSetup_cfi import *
process.trackerAlignment = cms.ESSource(
    "PoolDBESSource",
    CondDBSetup,
    connect = cms.string("$DB_PATH_TAG2"),
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
