#!/bin/bash

numFiles=15
cumRootFile='prova.root'
cumPBFile='prova.pb'
cumPBFile_inROOT='provaPB.root'
cumPBFileThreaded='provaThreaded.pb'
cumPBFileThreaded_inROOT='provaPBThreaded.root'
numThreads=3
timecmd='/usr/bin/time -f %E'

set_up() {
    echo "Removing previous ROOT and PB files"

    rm -fr fastHaddTests
    mkdir fastHaddTests
    cd fastHaddTests

    return 0
}

generate() {
    echo "Generating files"

    python ${LOCAL_TEST_DIR}/test_fastHaddMerge.py -a produce -n $numFiles 2>&1 > /dev/null

    if [ $? -ne 0 ]; then
        exit $?
    fi

    return 0
}

convertROOT2PB() {
    echo "Converting files to PB format"

    for file in $(ls Merge*root)
    do
        fastHadd encode -o `basename $file .root`.pb $file
#       cmsRun convertRoot2PB.py $file &> /dev/null
        if [ $? -ne 0 ]; then
            exit $?
        fi
    done

    return 0
}

hadd_merge() {
    echo "Merging with hadd"

    if [ -e "$cumRootFile" ]; then
        rm $cumRootFile
    fi

    $timecmd hadd $cumRootFile $(ls Merge*.root) 2>&1 > /dev/null

    if [ $? -ne 0 ]; then
        exit $?
    fi

    return 0
}

check_hadd() {
    echo "Checking ROOT result..."

    python ${LOCAL_TEST_DIR}/test_fastHaddMerge.py -a check -n $numFiles -c $cumRootFile 2>&1 > /dev/null

    if [ $? -ne 0 ]; then
        exit $?
    fi

    return 0
}

fasthadd_merge() {
    echo "Merging with fastHadd"

    $timecmd fastHadd -d add -o $cumPBFile $(ls Merge*.pb) 2>&1 > /dev/null

    if [ $? -ne 0 ]; then
        exit $?
    fi

    return 0
}

convertPB2ROOT() {
    echo "Converting back to ROOT format"

    fastHadd -d convert -o $cumPBFile_inROOT $cumPBFile 2>&1 > /dev/null

    if [ $? -ne 0 ]; then
        exit $?
    fi

    return 0
}

check_fasthadd() {
    echo "Checking PB result... on ${cumPBFile_inROOT}"

    python ${LOCAL_TEST_DIR}/test_fastHaddMerge.py -a check -n $numFiles -c $cumPBFile_inROOT 2>&1 > /dev/null

    if [ $? -ne 0 ]; then
        exit $?
    fi

    return 0
}

fasthadd_parallel_merge() {
    echo "Merging with parallel fastHadd, $numThreads threads"

    $timecmd fastHadd -d add -j $numThreads -o $cumPBFileThreaded $(ls Merge*.pb) 2>&1 > /dev/null

    if [ $? -ne 0 ]; then
        exit $?
    fi

    return 0
}

convert() {
    echo "Converting back to ROOT format"

    fastHadd -d convert -o $cumPBFileThreaded_inROOT $cumPBFileThreaded 2>&1 > /dev/null

    if [ $? -ne 0 ]; then
        exit $?
    fi
}

check_fasthadd_parallel() {
    echo "Checking PB result... on ${cumPBFileThreaded_inROOT}"

    python ${LOCAL_TEST_DIR}/test_fastHaddMerge.py -a check -n $numFiles -c $cumPBFileThreaded_inROOT 2>&1 > /dev/null

    if [ $? -ne 0 ]; then
        exit $?
    fi

    return 0
}

set_up

generate
convertROOT2PB
hadd_merge
check_hadd
fasthadd_merge
convertPB2ROOT
check_fasthadd
fasthadd_parallel_merge
convert
check_fasthadd_parallel

# Local Variables:
# show-trailing-whitespace: t
# truncate-lines: t
# End:
