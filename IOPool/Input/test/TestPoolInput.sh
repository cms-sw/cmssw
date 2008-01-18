#!/bin/sh
# Pass in name and status
function die { echo $1: status $2 ;  exit $2; }

rm -f ${LOCAL_TMP_DIR}/PoolInputTest.root ${LOCAL_TMP_DIR}/PoolInputOther.root
rm -f ${LOCAL_TMP_DIR}/PrePoolInputTest.cfg ${LOCAL_TMP_DIR}/PoolInputTest.cfg

cat > ${LOCAL_TMP_DIR}/PrePoolInputTest.cfg << !
# Configuration file for PrePoolInputTest 
process TESTPROD = {
	untracked PSet maxEvents = {untracked int32 input = 11}
	include "FWCore/Framework/test/cmsExceptionsFatal.cff"
	path p = {Thing}
	module Thing = ThingProducer {untracked int32 debugLevel = 1}
	module output = PoolOutputModule {
		untracked string fileName = '${LOCAL_TMP_DIR}/PoolInputTest.root'
	}
	source = EmptySource {
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
	untracked PSet maxEvents = {untracked int32 input = -1}
	include "FWCore/Framework/test/cmsExceptionsFatal.cff"
	path p = {OtherThing, Analysis}
	module OtherThing = OtherThingProducer {untracked int32 debugLevel = 1}
	module Analysis = OtherThingAnalyzer {untracked int32 debugLevel = 1}
	source = PoolSource {
		untracked uint32 setRunNumber = 621
		untracked vstring fileNames = {
			'file:${LOCAL_TMP_DIR}/PoolInputTest.root',
			'file:${LOCAL_TMP_DIR}/PoolInputOther.root'
		}
	}
}
!
cmsRun --parameter-set ${LOCAL_TMP_DIR}/PoolInputTest.cfg || die 'Failure using PoolInputTest.cfg' $?

#
# Test storing OtherThing as well
#

rm -f ${LOCAL_TMP_DIR}/PoolInputTest.root ${LOCAL_TMP_DIR}/PoolInputOtherThing.root
rm -f ${LOCAL_TMP_DIR}/PrePoolInputTest2.cfg ${LOCAL_TMP_DIR}/PoolInputTest2.cfg

cat > ${LOCAL_TMP_DIR}/PrePoolInputTest2.cfg << !
# Configuration file for PrePoolInputTest 
process TESTPROD = {
	untracked PSet maxEvents = {untracked int32 input = 11}
	include "FWCore/Framework/test/cmsExceptionsFatal.cff"
	path p = {Thing,OtherThing}
	module Thing = ThingProducer {untracked int32 debugLevel = 1}
	module output = PoolOutputModule {
		untracked string fileName = '${LOCAL_TMP_DIR}/PoolInputTest.root'
	}
	module output2 = PoolOutputModule {
		untracked string fileName = '${LOCAL_TMP_DIR}/PoolInputDropTest.root'
		untracked vstring outputCommands = {
		  "keep *",
		  "drop *_Thing_*_*"
		}
	}
	module OtherThing = OtherThingProducer {untracked int32 debugLevel = 1}
	source = EmptySource {
		untracked uint32 firstRun = 561
		untracked uint32 firstLuminosityBlock = 6
		untracked uint32 numberEventsInLuminosityBlock = 3
		untracked uint32 numberEventsInRun = 7
	}
        endpath ep = {output, output2}
}
!
cmsRun --parameter-set ${LOCAL_TMP_DIR}/PrePoolInputTest2.cfg || die 'Failure using PrePoolInputTest2.cfg' $?

cp ${LOCAL_TMP_DIR}/PoolInputTest.root ${LOCAL_TMP_DIR}/PoolInputOther.root

cat > ${LOCAL_TMP_DIR}/PoolInputTest2.cfg << !
# Configuration file for PoolInputTest
process TESTRECO = {
	untracked PSet maxEvents = {untracked int32 input = -1}
	include "FWCore/Framework/test/cmsExceptionsFatal.cff"
	path p = {Analysis}
	module Analysis = OtherThingAnalyzer {untracked int32 debugLevel = 1}
	source = PoolSource {
		untracked uint32 setRunNumber = 621
		untracked vstring fileNames = {
			'file:${LOCAL_TMP_DIR}/PoolInputTest.root',
			'file:${LOCAL_TMP_DIR}/PoolInputOther.root'
		}
	}
}
!
cmsRun --parameter-set ${LOCAL_TMP_DIR}/PoolInputTest2.cfg || die 'Failure using PoolInputTest2.cfg' $?

cat > ${LOCAL_TMP_DIR}/PoolInputTest3.cfg << !
# Configuration file for PoolInputTest
process TESTRECO = {
	untracked PSet maxEvents = {untracked int32 input = -1}
	include "FWCore/Framework/test/cmsExceptionsFatal.cff"
	path p = {Analysis}
	module Analysis = OtherThingAnalyzer {
		untracked int32 debugLevel = 1
		untracked bool thingWasDropped = true
	}
	source = PoolSource {
		untracked uint32 setRunNumber = 621
		untracked vstring fileNames = {
			'file:${LOCAL_TMP_DIR}/PoolInputDropTest.root'
		}
	}
}
!
cmsRun --parameter-set ${LOCAL_TMP_DIR}/PoolInputTest3.cfg || die 'Failure using PoolInputTest3.cfg' $?


cat > ${LOCAL_TMP_DIR}/PoolEmptyTest.cfg << !
# Configuration file for PoolInputTest
process WRITEEMPTY = {
	untracked PSet maxEvents = {untracked int32 input = -1}
	include "FWCore/Framework/test/cmsExceptionsFatal.cff"
	path p = {Thing}
	module Thing = ThingProducer {untracked int32 debugLevel = 1}
	module output = PoolOutputModule {
		untracked string fileName = '${LOCAL_TMP_DIR}/PoolEmptyTest.root'
	}
	source = TestRunLumiSource {
	      untracked vint32 runLumiEvent = {0, 0, 0}
	}
        endpath ep = {output}
}
!
cmsRun --parameter-set ${LOCAL_TMP_DIR}/PoolEmptyTest.cfg || die 'Failure using PoolEmptyTest.cfg' $?

cat > ${LOCAL_TMP_DIR}/PoolEmptyTest2.cfg << !
# Configuration file for PoolInputTest
process READEMPTY = {
	untracked PSet maxEvents = {untracked int32 input = -1}
	include "FWCore/Framework/test/cmsExceptionsFatal.cff"
	path p = {Thing}
	module Thing = ThingProducer {untracked int32 debugLevel = 1}
	module output = PoolOutputModule {
		untracked string fileName = '${LOCAL_TMP_DIR}/PoolEmptyTestOut.root'
	}
	source = PoolSource {
		untracked vstring fileNames = {'file:${LOCAL_TMP_DIR}/PoolEmptyTest.root'}
	}
        endpath ep = {output}
}
!
cmsRun --parameter-set ${LOCAL_TMP_DIR}/PoolEmptyTest2.cfg || die 'Failure using PoolEmptyTest2.cfg' $?
