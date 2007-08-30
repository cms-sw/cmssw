#!/bin/sh
# Pass in name and status
function die { echo $1: status $2 ;  exit $2; }

rm -f ${LOCAL_TMP_DIR}/PoolOutputTest.root
rm -f ${LOCAL_TMP_DIR}/PoolOutputTest.cfg
rm -f ${LOCAL_TMP_DIR}/PoolOutputRead.cfg
rm -f ${LOCAL_TMP_DIR}/PoolDropTest.root
rm -f ${LOCAL_TMP_DIR}/PoolDropTest.cfg
rm -f ${LOCAL_TMP_DIR}/PoolDropRead.cfg
rm -f ${LOCAL_TMP_DIR}/PoolMissingTest.root
rm -f ${LOCAL_TMP_DIR}/PoolMissingTest.cfg
rm -f ${LOCAL_TMP_DIR}/PoolMissingRead.cfg


cat > ${LOCAL_TMP_DIR}/PoolOutputTest.cfg << !
# Configuration file for PoolOutputTest
process TESTOUTPUT = {
	untracked PSet maxEvents = {untracked int32 input = 20}
	include "FWCore/Framework/test/cmsExceptionsFatal.cff"
	path p = {Thing, OtherThing}
	module Thing = ThingProducer {untracked int32 debugLevel = 1}
	module OtherThing = OtherThingProducer {untracked int32 debugLevel = 1}
	module output = PoolOutputModule {
		untracked string fileName = '${LOCAL_TMP_DIR}/PoolOutputTest.root'
		untracked int32 maxSize = 100000
	}
	source = EmptySource {}
	endpath ep = {output}
}
!
cmsRun --parameter-set ${LOCAL_TMP_DIR}/PoolOutputTest.cfg || die 'Failure using PoolOutputTest.cfg' $?

cat > ${LOCAL_TMP_DIR}/PoolDropTest.cfg << !
# Configuration file for PoolDropTest
process TESTDROP = {
	untracked PSet maxEvents = {untracked int32 input = 20}
	include "FWCore/Framework/test/cmsExceptionsFatal.cff"
	path p = {Thing, OtherThing}
	module Thing = ThingProducer {untracked int32 debugLevel = 1}
	module OtherThing = OtherThingProducer {untracked int32 debugLevel = 1}
	module output = PoolOutputModule {
		untracked string fileName = '${LOCAL_TMP_DIR}/PoolDropTest.root'
		untracked int32 maxSize = 100000
		untracked vstring outputCommands = {'drop *',
		    'keep *_dummy_*_*'
		    }
	}
	source = EmptySource {}
	endpath ep = {output}
}
!
cmsRun --parameter-set ${LOCAL_TMP_DIR}/PoolDropTest.cfg || die 'Failure using PoolDropTest.cfg' $?

cat > ${LOCAL_TMP_DIR}/PoolMissingTest.cfg << !
# Configuration file for PoolMissingTest
process TESTMISSING = {
	untracked PSet maxEvents = {untracked int32 input = 20}
	include "FWCore/Framework/test/cmsExceptionsFatal.cff"
	path p = {Thing}
	module Thing = ThingProducer {untracked bool noPut = true}
	module output = PoolOutputModule {
		untracked string fileName = '${LOCAL_TMP_DIR}/PoolMissingTest.root'
		untracked int32 maxSize = 100000
	}
	source = EmptySource {}
	endpath ep = {output}
}
!
cmsRun --parameter-set ${LOCAL_TMP_DIR}/PoolMissingTest.cfg || die 'Failure using PoolMissingTest.cfg' $?

cat > ${LOCAL_TMP_DIR}/PoolOutputRead.cfg << !
# Configuration file for PoolOutputRead
process TESTOUTPUTREAD = {
        untracked PSet maxEvents = {untracked int32 input = -1}
        include "FWCore/Framework/test/cmsExceptionsFatal.cff"
        source = PoolSource {
                untracked vstring fileNames = {
                        'file:${LOCAL_TMP_DIR}/PoolOutputTest.root'
                }
        }
}
!
cmsRun --parameter-set ${LOCAL_TMP_DIR}/PoolOutputRead.cfg || die 'Failure using PoolOutputRead.cfg' $?

cat > ${LOCAL_TMP_DIR}/PoolDropRead.cfg << !
# Configuration file for PoolDropRead
process TESTDROPREAD = {
        untracked PSet maxEvents = {untracked int32 input = -1}
        include "FWCore/Framework/test/cmsExceptionsFatal.cff"
        source = PoolSource {
                untracked vstring fileNames = {
                        'file:${LOCAL_TMP_DIR}/PoolDropTest.root'
                }
        }
}
!
cmsRun --parameter-set ${LOCAL_TMP_DIR}/PoolDropRead.cfg || die 'Failure using PoolDropRead.cfg' $?

cat > ${LOCAL_TMP_DIR}/PoolMissingRead.cfg << !
# Configuration file for PoolMissingRead
process TESTDROPREAD = {
        untracked PSet maxEvents = {untracked int32 input = -1}
        include "FWCore/Framework/test/cmsExceptionsFatal.cff"
        source = PoolSource {
                untracked vstring fileNames = {
                        'file:${LOCAL_TMP_DIR}/PoolMissingTest.root'
                }
        }
}
!
cmsRun --parameter-set ${LOCAL_TMP_DIR}/PoolMissingRead.cfg || die 'Failure using PoolMissingRead.cfg' $?

