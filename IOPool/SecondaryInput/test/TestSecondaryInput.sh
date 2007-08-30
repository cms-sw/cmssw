#!/bin/sh
# Pass in name and status
function die { echo $1: status $2 ;  exit $2; }

rm -f ${LOCAL_TMP_DIR}/SecondaryInputTest2.root ${LOCAL_TMP_DIR}/SecondaryInputTest.root ${LOCAL_TMP_DIR}/SecondaryInputOther.root
rm -f ${LOCAL_TMP_DIR}/PreSecondaryInputTest2.cfg ${LOCAL_TMP_DIR}/PreSecondaryInputTest.cfg ${LOCAL_TMP_DIR}/SecondaryInputTest.cfg

cat > ${LOCAL_TMP_DIR}/PreSecondaryInputTest2.cfg << !
# Configuration file for PreSecondaryInputTest2
process TEST = {
	untracked PSet maxEvents = {untracked int32 input = 5}
        include "FWCore/Framework/test/cmsExceptionsFatal.cff"
	path p = {Thing, OtherThing}
	module Thing = ThingProducer {untracked int32 debugLevel = 1}
	module OtherThing = OtherThingProducer {untracked int32 debugLevel = 1}
	module output = PoolOutputModule {
		untracked string fileName = '${LOCAL_TMP_DIR}/SecondaryInputTest2.root'
		untracked int32 maxSize = 100000
	}
	source = EmptySource {}
	endpath ep = {output}
}
!
cmsRun --parameter-set ${LOCAL_TMP_DIR}/PreSecondaryInputTest2.cfg || die 'Failure using PreSecondaryInputTest2.cfg' $?

cat > ${LOCAL_TMP_DIR}/PreSecondaryInputTest.cfg << !
# Configuration file for PreSecondaryInputTest 
process TEST = {
	untracked PSet maxEvents = {untracked int32 input = 50}
        include "FWCore/Framework/test/cmsExceptionsFatal.cff"
	path p = {Thing, OtherThing}
	module Thing = ThingProducer {untracked int32 debugLevel = 1}
	module OtherThing = OtherThingProducer {untracked int32 debugLevel = 1}
	module output = PoolOutputModule {
		untracked string fileName = '${LOCAL_TMP_DIR}/SecondaryInputTest.root'
		untracked int32 maxSize = 100000
	}
	source = EmptySource {}
	endpath ep = {output}
}
!
cmsRun --parameter-set ${LOCAL_TMP_DIR}/PreSecondaryInputTest.cfg || die 'Failure using PreSecondaryInputTest.cfg' $?

cat > ${LOCAL_TMP_DIR}/SecondaryInputTest.cfg << !
# Configuration file for SecondaryInputTest
process PROD  = {
	untracked PSet maxEvents = {untracked int32 input = 42}
	service = RandomNumberGeneratorService {
		untracked uint32 sourceSeed = 98765
		PSet moduleSeeds = {
			untracked uint32 Thing = 12345
		}
	}
        include "FWCore/Framework/test/cmsExceptionsFatal.cff"
	source = PoolSource { 
		untracked vstring fileNames = {'file:${LOCAL_TMP_DIR}/SecondaryInputTest.root'}
	}
	module Thing = SecondaryProducer {
	       secsource input = PoolSource  {
			untracked vstring fileNames = {'file:${LOCAL_TMP_DIR}/SecondaryInputTest2.root'}
	       }
	}
	module Analysis = EventContentAnalyzer {untracked bool verbose = true}
	path p = { Thing, Analysis }
}
!

cmsRun --parameter-set ${LOCAL_TMP_DIR}/SecondaryInputTest.cfg || die 'Failure using SecondaryInputTest.cfg' $?
