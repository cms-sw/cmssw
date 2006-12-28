#!/bin/sh
# Pass in name and status
function die { echo $1: status $2 ;  exit $2; }

rm -f ${LOCAL_TMP_DIR}/SecondaryInputTest2.root ${LOCAL_TMP_DIR}/SecondaryInputTest.root ${LOCAL_TMP_DIR}/SecondaryInputOther.root
rm -f ${LOCAL_TMP_DIR}/SecondaryInputTestCatalog.xml ${LOCAL_TMP_DIR}/SecondaryInputTestCatalog.xml.BAK
rm -f ${LOCAL_TMP_DIR}/PreSecondaryInputTest2.cfg ${LOCAL_TMP_DIR}/PreSecondaryInputTest.cfg ${LOCAL_TMP_DIR}/SecondaryInputTest.cfg

cat > ${LOCAL_TMP_DIR}/PreSecondaryInputTest2.cfg << !
# Configuration file for PreSecondaryInputTest2
process TEST = {
        include "FWCore/Framework/test/cmsExceptionsFatal.cff"
	path p = {Thing, OtherThing}
	module Thing = ThingProducer {untracked int32 debugLevel = 1}
	module OtherThing = OtherThingProducer {untracked int32 debugLevel = 1}
	module output = PoolOutputModule {
		untracked string fileName = '${LOCAL_TMP_DIR}/SecondaryInputTest2.root'
		untracked string catalog = '${LOCAL_TMP_DIR}/SecondaryInputTestCatalog.xml'
		untracked string logicalFileName = 'PoolTest2.root'
		untracked int32 maxSize = 100000
	}
	source = EmptySource {untracked int32 maxEvents = 5}
	endpath ep = {output}
}
!
cmsRun --parameter-set ${LOCAL_TMP_DIR}/PreSecondaryInputTest2.cfg || die 'Failure using PreSecondaryInputTest2.cfg' $?

cat > ${LOCAL_TMP_DIR}/PreSecondaryInputTest.cfg << !
# Configuration file for PreSecondaryInputTest 
process TEST = {
        include "FWCore/Framework/test/cmsExceptionsFatal.cff"
	path p = {Thing, OtherThing}
	module Thing = ThingProducer {untracked int32 debugLevel = 1}
	module OtherThing = OtherThingProducer {untracked int32 debugLevel = 1}
	module output = PoolOutputModule {
		untracked string fileName = '${LOCAL_TMP_DIR}/SecondaryInputTest.root'
		untracked string catalog = '${LOCAL_TMP_DIR}/SecondaryInputTestCatalog.xml'
		untracked string logicalFileName = 'PoolTest.root'
		untracked int32 maxSize = 100000
	}
	source = EmptySource {untracked int32 maxEvents = 50}
	endpath ep = {output}
}
!
cmsRun --parameter-set ${LOCAL_TMP_DIR}/PreSecondaryInputTest.cfg || die 'Failure using PreSecondaryInputTest.cfg' $?

cat > ${LOCAL_TMP_DIR}/SecondaryInputTest.cfg << !
# Configuration file for SecondaryInputTest
process PROD  = {
        include "FWCore/Framework/test/cmsExceptionsFatal.cff"
	source = PoolSource { 
		untracked int32 maxEvents = 42
		untracked string catalog = '${LOCAL_TMP_DIR}/SecondaryInputTestCatalog.xml'
		untracked vstring fileNames = {'PoolTest.root'}
	}
	module Thing = SecondaryProducer {
	       secsource input = PoolSource  {
			untracked string catalog = '${LOCAL_TMP_DIR}/SecondaryInputTestCatalog.xml'
			untracked vstring fileNames = {'PoolTest2.root'}
	       }
	}
	module Analysis = EventContentAnalyzer {untracked bool verbose = true}
	path p = { Thing, Analysis }
}
!

cmsRun --parameter-set ${LOCAL_TMP_DIR}/SecondaryInputTest.cfg || die 'Failure using SecondaryInputTest.cfg' $?
