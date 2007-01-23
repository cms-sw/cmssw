#!/bin/sh
# Pass in name and status
function die { echo $1: status $2 ;  exit $2; }

rm -f ${LOCAL_TMP_DIR}/PoolInputTest.root ${LOCAL_TMP_DIR}/PoolInputOther.root
rm -f ${LOCAL_TMP_DIR}/PoolInputTestCatalog.xml ${LOCAL_TMP_DIR}/PoolInputTestCatalog.xml.BAK
rm -f ${LOCAL_TMP_DIR}/PrePoolInputTest.cfg ${LOCAL_TMP_DIR}/PoolInputTest.cfg

cat > ${LOCAL_TMP_DIR}/PrePoolInputTest.cfg << !
# Configuration file for PrePoolInputTest 
process TESTPROD = {
	include "FWCore/Framework/test/cmsExceptionsFatal.cff"
	path p = {Thing}
	module Thing = ThingProducer {untracked int32 debugLevel = 1}
	module output = PoolOutputModule {
		untracked string fileName = '${LOCAL_TMP_DIR}/PoolInputTest.root'
		untracked string catalog = '${LOCAL_TMP_DIR}/PoolInputTestCatalog.xml'
		untracked string logicalFileName = 'PoolTest.root'
		untracked int32 maxSize = 100000
	}
	source = EmptySource {
		untracked int32 maxEvents = 11
		untracked uint32 firstRun = 561
		untracked uint32 firstLuminosityBlock = 6
		untracked uint32 numberEventsInLuminosityBlock = 3
		untracked uint32 numberEventsInRun = 7
	}
        endpath ep = {output}
}
!
cmsRun --parameter-set ${LOCAL_TMP_DIR}/PrePoolInputTest.cfg || die 'Failure using PrePoolInputTest.cfg' $?

cp ${LOCAL_TMP_DIR}/PoolInputTest.root ${LOCAL_TMP_DIR}/PoolInputOther.root

cat > ${LOCAL_TMP_DIR}/PoolInputTest.cfg << !
# Configuration file for PoolInputTest
process TESTRECO = {
	include "FWCore/Framework/test/cmsExceptionsFatal.cff"
	path p = {OtherThing, Analysis}
	module OtherThing = OtherThingProducer {untracked int32 debugLevel = 1}
	module Analysis = OtherThingAnalyzer {untracked int32 debugLevel = 1}
	source = PoolSource {
		untracked vstring fileNames = {
			'file:${LOCAL_TMP_DIR}/PoolInputTest.root',
			'PoolTest.root',
			'file:${LOCAL_TMP_DIR}/PoolInputOther.root'
		}
		untracked string catalog = '${LOCAL_TMP_DIR}/PoolInputTestCatalog.xml'
		untracked int32 maxEvents = -1
	}
}
!
cmsRun --parameter-set ${LOCAL_TMP_DIR}/PoolInputTest.cfg || die 'Failure using PoolInputTest.cfg' $?

