#!/bin/bash

set -e

if [ -z "$1" ]; then
	echo "No argument supplied"
	echo "Usage: $0 CMSSW_SOURCE_DIRECTORY"
	exit 1
fi

test -d ~/rpmbuild || ( echo "Please run rpmdev-setuptree first"; exit 1 )

TWD=`pwd`
TMP=`mktemp -d /tmp/rpmbuild.XXXXXXXXXX`
CMSSWSRC=`readlink -f "$1"`
COMMIT=`git log -n 1 --pretty=format:"%H"`
SOURCETAR=$TMP/fasthadd-$COMMIT.tar.gz
SPEC=$TMP/fasthadd.spec

echo $TMP
echo $CMSSWSRC
echo $COMMIT

mkdir -p $TMP/fasthadd-$COMMIT

# copy the spec file
cp -vp $TWD/fasthadd.spec $SPEC
sed -i -e s#COMMITHASH#$COMMIT# $SPEC

# create a source archive
cd $TMP/fasthadd-$COMMIT

cp -vp \
	$CMSSWSRC/DQMServices/Components/bin/fastHadd.cc \
	$CMSSWSRC/DQMServices/Core/src/ROOTFilePB.proto \
	$CMSSWSRC/DQMServices/Components/test/test_fastHaddMerge.py \
	$CMSSWSRC/DQMServices/Components/test/fastParallelHadd.py \
	$CMSSWSRC/DQMServices/Components/test/run_fastHadd_tests.sh \
	.

sed -i -e s#DQMServices/Core/src/ROOTFilePB.pb.h#ROOTFilePB.pb.h# fastHadd.cc
sed -i -e s#\$\{LOCAL_TEST_DIR\}/##g run_fastHadd_tests.sh

cd ..
tar czvf $SOURCETAR fasthadd-$COMMIT

# build stuff
echo $SPEC
rpmlint $SPEC

mkdir ~/rpmbuild/SPECS/ ~/rpmbuild/SOURCES/ || true

cp -vp $SPEC ~/rpmbuild/SPECS/
cp -vp $SOURCETAR ~/rpmbuild/SOURCES/
cd ~/rpmbuild/SPECS/
rpmbuild -ba -vv fasthadd.spec

rm -fr $TMP
