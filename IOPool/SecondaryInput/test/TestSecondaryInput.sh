#!/bin/sh
# Pass in name and status
function die { echo $1: status $2 ;  exit $2; }

rm -f ${LOCAL_TMP_DIR}/SecondaryInputTest.root ${LOCAL_TMP_DIR}/SecondaryInputOther.root
rm -f ${LOCAL_TMP_DIR}/SecondaryInputTestCatalog.xml ${LOCAL_TMP_DIR}/SecondaryInputTestCatalog.xml.BAK
rm -f ${LOCAL_TMP_DIR}/PreSecondaryInputTest.cfg ${LOCAL_TMP_DIR}/SecondaryInputTest.cfg

cat > ${LOCAL_TMP_DIR}/PreSecondaryInputTest.cfg << !
# Configuration file for PreSecondaryInputTest 
process TEST = {
	path p = {Thing, OtherThing, output}
	module Thing = ThingProducer {untracked int32 debugLevel = 1}
	module OtherThing = OtherThingProducer {untracked int32 debugLevel = 1}
	module output = PoolOutputModule {
		untracked string fileName = '${LOCAL_TMP_DIR}/SecondaryInputTest.root'
		untracked string catalog = '${LOCAL_TMP_DIR}/SecondaryInputTestCatalog.xml'
		untracked string logicalFileName = 'PoolTest.root'
		untracked int32 maxSize = 100000
	}
	source = EmptySource {untracked int32 maxEvents = 5}
}
!
cmsRun --parameter-set ${LOCAL_TMP_DIR}/PreSecondaryInputTest.cfg || die 'Failure using PreSecondaryInputTest.cfg' $?

cat > ${LOCAL_TMP_DIR}/SecondaryInputTest.cfg << !
# Configuration file for SecondaryInputTest
process PROD  = {
	source = EmptySource { 
		untracked int32 maxEvents = 42
	}
	module Thing = SecondaryProducer {
	       secsource input = PoolRASource  {
			untracked string catalog = '${LOCAL_TMP_DIR}/SecondaryInputTestCatalog.xml'
			untracked vstring fileNames = {'PoolTest.root'}
	       }
	}
	module Analysis = EventContentAnalyzer {untracked bool verbose = true}
	path p = { Thing, Analysis }
}
!

cmsRun --parameter-set ${LOCAL_TMP_DIR}/SecondaryInputTest.cfg || die 'Failure using SecondaryInputTest.cfg' $?
