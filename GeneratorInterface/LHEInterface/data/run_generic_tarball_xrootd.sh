#!/bin/bash

set -e

echo "   ______________________________________     "

if [ $# -lt 1 ]; then
    echo "%MSG-ExternalLHEProducer-subprocess ERROR in external process. The gridpack path must be passed as an argument"
fi
if [[ $1 != "root://"* ]]; then
    echo "%MSG-ExternalLHEProducer-subprocess ERROR in external process. Path must have format root://<xrd_path>/<path>"
    exit 1
fi 

xrd_path=$1
gridpack=$(basename $xrd_path)

if [ -e $gridpack ]; then
    echo "%MSG-ExternalLHEProducer-subprocess WARNING: File $gridpack already exists, it will be overwritten."
    rm $gridpack
fi

echo "%MSG-ExternalLHEProducer-subprocess INFO: Copying gridpack $xrd_path locally using xrootd"
xrdcp $xrd_path .

path=`pwd`/$gridpack
generic_script=$(dirname ${0})/run_generic_tarball_cvmfs.sh 
. $generic_script $path ${@:2}
