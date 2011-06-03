#!/bin/bash
#set -xv

if [ $# -lt 1 ]; then
  echo "Usage: $0  [tar-dir]"
  exit 1
fi

# TODO: add 'std' option so file name is automatically created from release name and date

tard=${PWD}/$1
if [ -f $tard ]; then
   echo "dir exist alreday"
   exit
fi
mkdir $tard
echo "Info: set $tard for destination"

origd=$PWD;
#----------------------------------------------------------------
# root
# can be linked or installed at $ROOTSYS

mkdir  ${tard}/external

ROOTSYS=`echo $ROOTSYS |  sed 's/\/$//'` # remove '/' character  at end of string, becuse it invalidates symblic link interpretation
origr=$ROOTSYS
if [ -L ${ROOTSYS} ]; then
   b=`dirname ${ROOTSYS}`
   origr=${b}/`readlink ${ROOTSYS}`
fi

echo "copy root from $origr to ${tard}/external/root"
cp -a $origr  ${tard}/external/root
#exit
#----------------------------------------------------------------
# external libraries
extdl=${tard}/external/lib
mkdir $extdl

ext=`dirname ${CMSSW_DATA_PATH}`/external

echo "Copying external libraries from $ext to $extdl."

# cp -a $ext/*/*/lib/*  ${tard}/external/lib
for i in boost bz2lib castor clhep dcap db4 dcap \
	elementtree expat fftw3 gdbm gsl hepmc\
	libjpg libpng libtiff libungif \
	openldap openssl pcre \
	sigcpp sqlite xrootd zlib
do
	echo "cp -a $ext/$i/*/lib/* === ${extdl}"
	cp -a $ext/$i/*/lib/* ${extdl}
done

#----------------------------------------------------------------
# cmssw
   
mkdir -p ${tard}/lib

# cms libraries
cp -a $CMSSW_RELEASE_BASE/lib/*/* ${tard}/lib/
cp -a $CMSSW_BASE/lib/*/* ${tard}/lib/

# plugins cache file
touch ${tard}/lib/.edmplugincache
cat  $CMSSW_RELEASE_BASE/lib/*/.edmplugincache > ${tard}/lib/.edmplugincache
echo "get $CMSSW_RELEASE_BASE/lib/*/.edmplugincache > ${tard}/lib/.edmplugincache"
# cat  $CMSSW_BASE/lib/*/.edmplugincache >> ${tard}/lib/.edmplugincache

#----------------------------------------------------------------

# binary 
cp $CMSSW_BASE/bin/*/cmsShow.exe ${tard}

# src
srcDir="${tard}/src/Fireworks/Core"
mkdir -p $srcDir
cp -a  $CMSSW_BASE/src/Fireworks/Core/macros $srcDir
cp -a  $CMSSW_BASE/src/Fireworks/Core/icons $srcDir
cp -a  $CMSSW_BASE/src/Fireworks/Core/data $srcDir
cp -a  $CMSSW_BASE/src/Fireworks/Core/scripts/cmsShow $tard

ln -s  $CMSSW_BASE/src/Fireworks/Core/macros/default.fwc  $tard
ln -s  $CMSSW_BASE/src/Fireworks/Core/macros/ispy.fwc  $tard
ln -s  $CMSSW_BASE/src/Fireworks/Core/macros/pflow.fwc  $tard
ln -s  $CMSSW_BASE/src/Fireworks/Core/macros/hfLego.fwc  $tard

cp  $CMSSW_DATA_PATH/data-Fireworks-Geometry/4-cms/Fireworks/Geometry/data/* .

#----------------------------------------------------------------

# sample files
cd $tard
dwnCmd="wget"
if [ `uname` = "Darwin" ]; then
   # compatibility problem on 10.5
   dwnCmd="curl -O"
fi
cd ${tard}
name=`perl -e '($ver, $a, $b, $c) = split('_', $ENV{CMSSW_VERSION}); print  "data", $a, $b, ".root"  '`
$dwnCmd http://amraktad.web.cern.ch/amraktad/mail/scratch0/data/$name
mv $name data.root


cd $origd
echo "Creating tarball ..."
if [ `uname` = "Darwin" ]; then
    echo "tar -czf ${tard}.mac.tar.gz $tard"
else
    echo "tar -czf ${tard}.linux.tar.gz $tard"