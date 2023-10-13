#!/bin/sh -x

function die { echo $1: status $2 ;  exit $2; }

LOCAL_TEST_DIR=${SCRAM_TEST_PATH}

# The purpose of this is to test schema evolution in ROOT.

# In the first two cmsRun processes, there is no schema evolution
# going on. The first one writes a data file containing a test
# product. The second one reads the test products and checks
# that the values read from it match values that the first process
# should have written. The purpose here is to validate that the
# test code is actually working properly and if we see failures
# in the later cmsRun processes, they are more likely to be
# caused by a failure in ROOT schema evolution.
cmsRun ${LOCAL_TEST_DIR}/SchemaEvolution_create_test_file_cfg.py || die 'Failure using SchemaEvolution_create_test_file_cfg.py' $?
cmsRun ${LOCAL_TEST_DIR}/SchemaEvolution_test_read_cfg.py || die 'Failure using SchemaEvolution_create_test_file_cfg.py' $?

# For each StreamerInfo in the input file, test for existence of StreamerInfo for
# nested classes (members, base, elements of containers).
root.exe -b -l -q file:SchemaEvolutionTest.root "${LOCAL_TEST_DIR}/testForStreamerInfo.C(gFile)" | sort -u > testForStreamerInfo1.log
grep "Missing" testForStreamerInfo1.log && die "Missing nested streamer info" 1
grep "SchemaEvolutionChangeOrder" testForStreamerInfo1.log || die 'Failure cannot find SchemaEvolutionChangeOrder in testForStreamerInfo1.log' $?
grep "SchemaEvolutionAddMember" testForStreamerInfo1.log || die 'Failure cannot find SchemaEvolutionAddMember in testForStreamerInfo1.log' $?
grep "SchemaEvolutionRemoveMember" testForStreamerInfo1.log || die 'Failure cannot find SchemaEvolutionRemoveMember in testForStreamerInfo1.log' $?
grep "SchemaEvolutionMoveToBase" testForStreamerInfo1.log || die 'Failure cannot find SchemaEvolutionMoveToBase" in testForStreamerInfo1.log' $?
grep "SchemaEvolutionChangeType" testForStreamerInfo1.log || die 'Failure cannot find SchemaEvolutionChangeType in testForStreamerInfo1.log' $?
grep "SchemaEvolutionAddBase" testForStreamerInfo1.log || die 'Failure cannot find SchemaEvolutionAddBase in testForStreamerInfo1.log' $?
grep "SchemaEvolutionPointerToMember" testForStreamerInfo1.log || die 'Failure cannot find SchemaEvolutionPointerToMember in testForStreamerInfo1.log' $?
grep "SchemaEvolutionPointerToUniquePtr" testForStreamerInfo1.log || die 'Failure cannot find SchemaEvolutionPointerToUniquePtr in testForStreamerInfo1.log' $?
grep "SchemaEvolutionCArrayToStdArray" testForStreamerInfo1.log || die 'Failure cannot find SchemaEvolutionCArrayToStdArray in testForStreamerInfo1.log' $?
grep "SchemaEvolutionVectorToList" testForStreamerInfo1.log || die 'Failure cannot find SchemaEvolutionVectorToList in testForStreamerInfo1.log' $?
grep "SchemaEvolutionMapToUnorderedMap" testForStreamerInfo1.log || die 'Failure cannot find SchemaEvolutionMapToUnorderedMap in testForStreamerInfo1.log' $?
grep "VectorVectorElementNonSplit" testForStreamerInfo1.log || die 'Failure cannot find VectorVectorElementNonSplit in testForStreamerInfo1.log' $?

# Then we read permanently saved data files from the cms-data
# repository. When these data files were written, the working area
# was built with different class definitions used for the contents
# of the test product. This forces ROOT to perform schema evolution
# as it reads the test product. We check both that the job
# completes successfully and that the values contained in
# the test product after schema evolution are what we expect.
# The plan is to generate new data files each time there is a major
# revision to ROOT.

# The old files read below were generated as follows.
#
#     Check out the release you want to use to generate the file
#     and build a working area. You will at least need to checkout
#     IOPool/Input and DataFormats/TestObjects.
#
#     If the release is before the first release that includes this
#     file (testSchemaEvolution.sh), then you will need to manually
#     merge in the commit that added the file. Most of the files are
#     completely new files, although BuildFile.xml, classes.h, and
#     classes_def.xml already existed may require manually merging a
#     few lines of code.
#
#     Manually edit 2 files.
#
#     1. Modify the following line in the file:
#     DataFormats/TestObjects/interface/SchemaEvolutionTestObjects.h
#     //#define DataFormats_TestObjects_USE_OLD
#     To generate an input file with the old format remove the "//"
#     at the beginning so the macro is defined.
#
#     2. In file: DataFormats/TestObjects/src/classes_def.xml
#     There is a section of code starting with SchemaEvolutionChangeOrder
#     continuing to SchemaEvolutionMapToUnorderedMap. This section appears
#     twice. The first section defines old formats that should be used
#     to generate old format data files (These are all ClassVersion= "3")
#     When reading, use the new version (the section with ClassVersion="4"
#     which should be the one enabled in the CMSSW repository).
#     Of these two sections, exactly one should be commented out.
#     To generate the input data files, you will need to manually
#     enable the ClassVersion 3 section.
#
#     Then rebuild the working area.
#
#     Then run the configuration IOPool/Input/test/SchemaEvolution_create_test_file_cfg.py
#     This will generate an output file named SchemaEvolutionTest.root.
#     Rename this file appropriately to include the release used and use
#     it as an input for this test by adding additional cases below.
#     The new data file will need to added to the cms-data repository
#     named IOPool-Input.

file=SchemaEvolutionTestOLD13_2_3.root
inputfile=$(edmFileInPath IOPool/Input/data/$file) || die "Failure edmFileInPath IOPool/Input/data/$file" $?
cmsRun ${LOCAL_TEST_DIR}/SchemaEvolution_test_read_cfg.py --inputFile "$inputfile" || die "Failed to read old file $file" $?

file=SchemaEvolutionTestOLD13_0_0.root
inputfile=$(edmFileInPath IOPool/Input/data/$file) || die "Failure edmFileInPath IOPool/Input/data/$file" $?
# These fail because there was a bug in the version of ROOT associated with CMSSW_13_0_0
# The bug caused StreamerInfo objects to missing from the ROOT file. In this case,
# schema evolution fails and also the testForStreamerInfo.C script will find
# missing StreamerInfo objects.
# Lets keep this code around because it may be useful if we need to
# do additional work related to data files written using an executable
# built from code having the bug.
#cmsRun ${LOCAL_TEST_DIR}/SchemaEvolution_test_read_cfg.py --inputFile "$inputfile" || die "Failed to read old file $file" $?
#root.exe -b -l -q file:$inputfile "${LOCAL_TEST_DIR}/testForStreamerInfo.C(gFile)" | sort -u | grep Missing > testForStreamerInfo2.log
#grep "Missing" testForStreamerInfo2.log && die "Missing nested streamer info" 1

exit 0
