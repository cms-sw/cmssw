#!/bin/sh
# Pass in name and status
function die { echo $1: status $2 ;  exit $2; }

INPUT_1=${LOCAL_TMP_DIR}/FastMergeTest_1.root
INPUT_2=${LOCAL_TMP_DIR}/FastMergeTest_2.root
LOGICAL_INPUT_1=PoolTest_1.root
LOGICAL_INPUT_2=PoolTest_2.root

rm -f ${INPUT_1} ${INPUT_2}
rm -f ${LOCAL_TMP_DIR}/PreFastMergeTest_1.cfg          ${LOCAL_TMP_DIR}/PreFastMergeTest_2.cfg

#---------------------------
# Create first input file
#---------------------------

cat > ${LOCAL_TMP_DIR}/PreFastMergeTest_1.cfg << !
# Configuration file for PreFastMergeTest_1
process TESTPROD = {
	untracked PSet maxEvents = {untracked int32 input = 10}
        include "FWCore/Framework/test/cmsExceptionsFatal.cff"
	path p = {Thing, OtherThing}
	module Thing = ThingProducer {untracked int32 debugLevel = 0}
	module OtherThing = OtherThingProducer {untracked int32 debugLevel = 0}
	module output = PoolOutputModule {
		untracked string fileName = '${INPUT_1}'
                untracked vstring outputCommands = {
			"keep *",
			"drop *_OtherThing_*_*"
               }
	}
	source = EmptySource {
                 untracked uint32 firstEvent = 1
                 untracked uint32 firstRun = 100

        }
        endpath ep = {output}
}
!
cmsRun --parameter-set ${LOCAL_TMP_DIR}/PreFastMergeTest_1.cfg || die 'Failure using PreFastMergeTest_1.cfg' $?

#---------------------------
# Create second input file
#---------------------------

cat > ${LOCAL_TMP_DIR}/PreFastMergeTest_2.cfg << !
# Configuration file for PreFastMergeTest_2
process TESTPROD = {
	untracked PSet maxEvents = {untracked int32 input = 15}
        include "FWCore/Framework/test/cmsExceptionsFatal.cff"
	path p = {Thing, OtherOtherThing}
	module Thing = ThingProducer {untracked int32 debugLevel = 0}
	module OtherOtherThing = OtherThingProducer {untracked int32 debugLevel = 0}
	module output = PoolOutputModule {
		untracked string fileName = '${INPUT_2}'
                untracked vstring outputCommands = {
			"keep *",
			"drop *_OtherOtherThing_*_*"
               }
	}
	source = EmptySource {
                 untracked uint32 firstEvent = 100
                 untracked uint32 firstRun = 200
        }
        endpath ep = {output}
}
!
cmsRun --parameter-set ${LOCAL_TMP_DIR}/PreFastMergeTest_2.cfg || die 'Failure using PreFastMergeTest_2.cfg' $?


#---------------------------
# Merge files
#---------------------------

cat > ${LOCAL_TMP_DIR}/FastMergeTest.cfg << !
# Configuration file for FastMergeTest
process TESTMERGE = {
	service = AdaptorConfig{untracked bool stats = false}
	untracked PSet maxEvents = {untracked int32 input = -1}
        include "FWCore/Framework/test/cmsExceptionsFatal.cff"
	module output = PoolOutputModule {
		untracked string fileName = '${LOCAL_TMP_DIR}/FastMerge_out.root'
	}
	source = PoolSource {
		untracked vstring fileNames = {
			'file:${INPUT_1}',
			'file:${INPUT_2}'
		}
		untracked bool dropMetaData = true
        }
        endpath ep = {output}
}
!
cmsRun -j ${LOCAL_TMP_DIR}/TestFastMergeFJR.xml --parameter-set ${LOCAL_TMP_DIR}/FastMergeTest.cfg || die 'Failure using FastMergeTest.cfg' $?
#need to filter items in job report which always change
egrep -v "<GUID>|<PFN>" $LOCAL_TEST_DIR/proper_fjr_output > $LOCAL_TMP_DIR/proper_fjr_output
egrep -v "<GUID>|<PFN>" $LOCAL_TMP_DIR/TestFastMergeFJR.xml  > $LOCAL_TMP_DIR/TestFastMergeFJR_filtered.xml
diff $LOCAL_TMP_DIR/proper_fjr_output $LOCAL_TMP_DIR/TestFastMergeFJR_filtered.xml || die 'output framework job report is wrong' $?

