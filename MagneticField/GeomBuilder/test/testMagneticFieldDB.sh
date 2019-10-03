#!/bin/sh

LOCAL_TEST_DIR=src/MagneticField/GeomBuilder/test

function die { echo Failure $1: status $2 ; exit $2 ; }

cmsRun ${LOCAL_TEST_DIR}/python/testMagneticFieldDB_cfg.py  > testMagneticFieldDB.run_log || die "cmsRun testMagneticFieldDB_cfg.py" $?
diff testMagneticFieldDB.run_log ${LOCAL_TEST_DIR}/testMagneticFieldDB_cfg.results || die 'incorrect output using testMagneticFieldDB.run_log' $? 
rm testMagneticFieldDB.run_log

cmsRun ${LOCAL_TEST_DIR}/python/testMagneticFieldDB_valueOverride_cfg.py  > testMagneticFieldDB_valueOverride.run_log || die "cmsRun testMagneticFieldDB_valueOverride_cfg.py" $?
diff testMagneticFieldDB_valueOverride.run_log ${LOCAL_TEST_DIR}/testMagneticFieldDB_valueOverride_cfg.results || die 'incorrect output using testMagneticFieldDB_valueOverride.run_log' $? 
rm testMagneticFieldDB_valueOverride.run_log

cmsRun ${LOCAL_TEST_DIR}/python/testMagneticFieldDB_parameterized_cfg.py  > testMagneticFieldDB_parameterized.run_log || die "cmsRun testMagneticFieldDB_parameterized_cfg.py" $?
diff testMagneticFieldDB_parameterized.run_log ${LOCAL_TEST_DIR}/testMagneticFieldDB_parameterized_cfg.results || die 'incorrect output using testMagneticFieldDB_parameterized.run_log' $? 
rm testMagneticFieldDB_parameterized.run_log

