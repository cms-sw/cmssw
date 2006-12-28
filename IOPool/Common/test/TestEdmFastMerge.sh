#!/bin/sh
# Pass in name and status
function die { echo $1: status $2 ;  exit $2; }

INPUT_1=${LOCAL_TMP_DIR}/EdmFastMergeTest_1.root
INPUT_2=${LOCAL_TMP_DIR}/EdmFastMergeTest_2.root
LOGICAL_INPUT_1=PoolTest_1.root
LOGICAL_INPUT_2=PoolTest_2.root

rm -f ${INPUT_1} ${INPUT_2}
rm -f ${LOCAL_TMP_DIR}/EdmFastMergeTestCatalog.xml
rm -f ${LOCAL_TMP_DIR}/EdmFastMergeTestCatalog.xml.BAK
rm -f ${LOCAL_TMP_DIR}/PreEdmFastMergeTest_1.cfg          ${LOCAL_TMP_DIR}/PreEdmFastMergeTest_2.cfg

#---------------------------
# Create first input file
#---------------------------

cat > ${LOCAL_TMP_DIR}/PreEdmFastMergeTest_1.cfg << !
# Configuration file for PreEdmFastMergeTest_1
process TESTPROD = {
        include "FWCore/Framework/test/cmsExceptionsFatal.cff"
	path p = {Thing, OtherThing}
	module Thing = ThingProducer {untracked int32 debugLevel = 0}
	module OtherThing = OtherThingProducer {untracked int32 debugLevel = 0}
	module output = PoolOutputModule {
		untracked string fileName = '${INPUT_1}'
		untracked string catalog = '${LOCAL_TMP_DIR}/EdmFastMergeTestCatalog.xml'
		untracked string logicalFileName = '${LOGICAL_INPUT_1}'
		untracked int32 maxSize = 100000
	}
	source = EmptySource {
                 untracked int32 maxEvents = 10
                 untracked uint32 firstEvent = 1
                 untracked uint32 firstRun = 100

        }
        endpath ep = {output}
}
!
cmsRun --parameter-set ${LOCAL_TMP_DIR}/PreEdmFastMergeTest_1.cfg || die 'Failure using PreEdmFastMergeTest_1.cfg' $?

#---------------------------
# Create second input file
#---------------------------

cat > ${LOCAL_TMP_DIR}/PreEdmFastMergeTest_2.cfg << !
# Configuration file for PreEdmFastMergeTest_2
process TESTPROD = {
        include "FWCore/Framework/test/cmsExceptionsFatal.cff"
	path p = {Thing, OtherThing}
	module Thing = ThingProducer {untracked int32 debugLevel = 0}
	module OtherThing = OtherThingProducer {untracked int32 debugLevel = 0}
	module output = PoolOutputModule {
		untracked string fileName = '${INPUT_2}'
		untracked string catalog = '${LOCAL_TMP_DIR}/EdmFastMergeTestCatalog.xml'
		untracked string logicalFileName = '${LOGICAL_INPUT_2}'
		untracked int32 maxSize = 100000
	}
	source = EmptySource {
                 untracked int32 maxEvents = 15
                 untracked uint32 firstEvent = 100
                 untracked uint32 firstRun = 200
        }
        endpath ep = {output}
}
!
cmsRun --parameter-set ${LOCAL_TMP_DIR}/PreEdmFastMergeTest_2.cfg || die 'Failure using PreEdmFastMergeTest_2.cfg' $?


#---------------------------
# Merge files
#---------------------------

EdmFastMerge -i file:${INPUT_1} ${LOGICAL_INPUT_2} -j ${LOCAL_TMP_DIR}/FrameworkJobReport.xml -o ${LOCAL_TMP_DIR}/EdmFastMerge_out.root -c ${LOCAL_TMP_DIR}/EdmFastMergeTestCatalog.xml || die 'Failure using EdmFastMerge' $?


