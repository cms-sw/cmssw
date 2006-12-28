#!/bin/sh
# Pass in name and status
function die { echo $1: status $2 ;  exit $2; }

rm -f ${LOCAL_TMP_DIR}/PoolOutputTest.root ${LOCAL_TMP_DIR}/PoolOutputTestCatalog.xml ${LOCAL_TMP_DIR}/PoolOutputTestCatalog.xml.BAK ${LOCAL_TMP_DIR}/PoolOutputTest.cfg

cat > ${LOCAL_TMP_DIR}/PoolOutputTest.cfg << !
# Configuration file for PoolOutputTest
process TESTPROD = {
	include "FWCore/Framework/test/cmsExceptionsFatal.cff"
	path p = {Thing, OtherThing}
	module Thing = ThingProducer {untracked int32 debugLevel = 1}
	module OtherThing = OtherThingProducer {untracked int32 debugLevel = 1}
	module output = PoolOutputModule {
		untracked string fileName = '${LOCAL_TMP_DIR}/PoolOutputTest.root'
		untracked string catalog = '${LOCAL_TMP_DIR}/PoolOutputTestCatalog.xml'
		untracked string logicalFileName = 'PoolTest.root'
		untracked int32 maxSize = 100000
	}
	source = EmptySource {untracked int32 maxEvents = 20}
	endpath ep = {output}
}
!
cmsRun --parameter-set ${LOCAL_TMP_DIR}/PoolOutputTest.cfg || die 'Failure using PoolOutputTest.cfg' $?
