#!/bin/sh

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
cmsRun ${LOCAL_TEST_DIR}/SchemaEvolution_create_test_file_cfg.py --splitLevel 0 || die 'Failure using SchemaEvolution_create_test_file_cfg.py splitLevel 0' $?
cmsRun ${LOCAL_TEST_DIR}/SchemaEvolution_test_read_cfg.py --inputFile SchemaEvolutionTest_splitLevel0.root --testAutoPtrToUniquePtr || die 'Failure using SchemaEvolution_create_test_file_cfg.py splitLevel 0' $?

cmsRun ${LOCAL_TEST_DIR}/SchemaEvolution_create_test_file_cfg.py --splitLevel 99 || die 'Failure using SchemaEvolution_create_test_file_cfg.py splitLevel 99' $?
cmsRun ${LOCAL_TEST_DIR}/SchemaEvolution_test_read_cfg.py --inputFile SchemaEvolutionTest_splitLevel99.root --testAutoPtrToUniquePtr || die 'Failure using SchemaEvolution_create_test_file_cfg.py splitLevel 99' $?


# For each StreamerInfo in the input file, test for existence of StreamerInfo for
# nested classes (members, base, elements of containers).
for SPLIT in 0 99; do
    root.exe -b -l -q file:SchemaEvolutionTest_splitLevel${SPLIT}.root "${LOCAL_TEST_DIR}/testForStreamerInfo.C(gFile)" | sort -u > testForStreamerInfo_${SPLIT}.log
    grep "Missing" testForStreamerInfo_${SPLIT}.log && die "Missing nested streamer info" 1
    grep "SchemaEvolutionChangeOrder" testForStreamerInfo_${SPLIT}.log || die 'Failure cannot find SchemaEvolutionChangeOrder in testForStreamerInfo_${SPLIT}.log' $?
    grep "SchemaEvolutionAddMember" testForStreamerInfo_${SPLIT}.log || die 'Failure cannot find SchemaEvolutionAddMember in testForStreamerInfo_${SPLIT}.log' $?
    grep "SchemaEvolutionRemoveMember" testForStreamerInfo_${SPLIT}.log || die 'Failure cannot find SchemaEvolutionRemoveMember in testForStreamerInfo_${SPLIT}.log' $?
    grep "SchemaEvolutionMoveToBase" testForStreamerInfo_${SPLIT}.log || die 'Failure cannot find SchemaEvolutionMoveToBase" in testForStreamerInfo_${SPLIT}.log' $?
    grep "SchemaEvolutionChangeType" testForStreamerInfo_${SPLIT}.log || die 'Failure cannot find SchemaEvolutionChangeType in testForStreamerInfo_${SPLIT}.log' $?
    grep "SchemaEvolutionAddBase" testForStreamerInfo_${SPLIT}.log || die 'Failure cannot find SchemaEvolutionAddBase in testForStreamerInfo_${SPLIT}.log' $?
    grep "SchemaEvolutionPointerToMember" testForStreamerInfo_${SPLIT}.log || die 'Failure cannot find SchemaEvolutionPointerToMember in testForStreamerInfo_${SPLIT}.log' $?
    grep "SchemaEvolutionPointerToUniquePtr" testForStreamerInfo_${SPLIT}.log || die 'Failure cannot find SchemaEvolutionPointerToUniquePtr in testForStreamerInfo_${SPLIT}.log' $?
    grep "SchemaEvolutionCArrayToStdArray" testForStreamerInfo_${SPLIT}.log || die 'Failure cannot find SchemaEvolutionCArrayToStdArray in testForStreamerInfo_${SPLIT}.log' $?
    grep "SchemaEvolutionVectorToList" testForStreamerInfo_${SPLIT}.log || die 'Failure cannot find SchemaEvolutionVectorToList in testForStreamerInfo_${SPLIT}.log' $?
    grep "SchemaEvolutionMapToUnorderedMap" testForStreamerInfo_${SPLIT}.log || die 'Failure cannot find SchemaEvolutionMapToUnorderedMap in testForStreamerInfo_${SPLIT}.log' $?
    grep "VectorVectorElementNonSplit" testForStreamerInfo_${SPLIT}.log || die 'Failure cannot find VectorVectorElementNonSplit in testForStreamerInfo_${SPLIT}.log' $?
done

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
#     twice: with arguments' --splitLevel 0' and '--splitLevel 99'
#     This will generate an output file named SchemaEvolutionTest_splitLevel<N>.root.
#     Rename this file appropriately to include the release used and use
#     it as an input for this test by adding additional cases below.
#     The new data file will need to added to the cms-data repository
#     named IOPool-Input.

file=SchemaEvolutionTestOLD15_1_0_pre5_splitLevel0.root
inputfile=$(edmFileInPath IOPool/Input/data/$file) || die "Failure edmFileInPath IOPool/Input/data/$file" $?
cmsRun ${LOCAL_TEST_DIR}/SchemaEvolution_test_read_cfg.py --inputFile "$inputfile" --testAutoPtrToUniquePtr || die "Failed to read old file $file" $?

file=SchemaEvolutionTestOLD15_1_0_pre5_splitLevel99.root
inputfile=$(edmFileInPath IOPool/Input/data/$file) || die "Failure edmFileInPath IOPool/Input/data/$file" $?
cmsRun ${LOCAL_TEST_DIR}/SchemaEvolution_test_read_cfg.py --inputFile "$inputfile" --testAutoPtrToUniquePtr || die "Failed to read old file $file" $?

file=SchemaEvolutionTestOLD13_0_0.root
inputfile=$(edmFileInPath IOPool/Input/data/$file) || die "Failure edmFileInPath IOPool/Input/data/$file" $?


## Note that the following two tests exercise implicitly only split level 99

# The next test demonstrates the FileReadError that can occur as a
# result of the known ROOT bug in 13_0_0 (file has a problem when
# written with 13_0_0 that causes an exception when read).
# Note that this is also used to test the cmsRun exit code
# after a FileReadError (should be 8021). It is very convenient
# to test that here because it is hard to intentionally create
# a file that will cause a FileReadError. So we take advantage
# of the ROOT bug to implement the test. This bug actually
# occurred, see Issue 42179 for details.
echo "***"
echo "***"
echo "Exception in next test is INTENTIONAL. Test fails if not thrown or cmsRun returns wrong exit code"
echo "***"
echo "***"
cmsRun  -j FileReadErrorTest_jobreport.xml ${LOCAL_TEST_DIR}/SchemaEvolution_test_read_cfg.py --inputFile $inputfile && die 'SchemaEvolution_test_read_cfg.py with corrupt input did not throw an exception' 1
CMSRUN_EXIT_CODE=$(edmFjrDump --exitCode FileReadErrorTest_jobreport.xml)
if [ "x${CMSRUN_EXIT_CODE}" != "x8021" ]; then
    echo "cmsRun reported exit code ${CMSRUN_EXIT_CODE} which is different from the expected 8021 (FileReadError)"
    exit 1
fi

# The test below would fail without the "--enableStreamerInfosFix"
# because there was a bug in the version of ROOT associated with CMSSW_13_0_0.
# The bug caused StreamerInfo objects to be missing from the ROOT file. In this case,
# schema evolution fails without the fix and also the testForStreamerInfo.C script will
# find missing StreamerInfo objects.
cmsRun ${LOCAL_TEST_DIR}/SchemaEvolution_test_read_cfg.py --inputFile $inputfile --enableStreamerInfosFix || die "Failed to read old file $file with fix" $?

exit 0
