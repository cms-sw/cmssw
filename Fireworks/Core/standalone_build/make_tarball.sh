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
echo "tar dir $tard"

#----------------------------------------------------------------
# root
# can be linked or installed at $ROOTSYS

mkdir  ${tard}/external

ROOTSYS=`echo $ROOTSYS |  sed 's/\/$//'` # remove '/' character  at end of string, becuse it invalidates symblic link interpretation
origr=$ROOTSYS
if [ -L ${ROOTSYS} ]; then
   b=${ROOTSYS}
   origr=`readlink ${ROOTSYS}`
fi

echo "copy root from $origr to ${tard}/external/root"
cp -a $origr  ${tard}/external/root

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

# binary 
cp $CMSSW_BASE/bin/*/cmsShow.exe ${tard}

# version file, icons, and configuration files
mkdir -p ${tard}/src
cvs co -p Fireworks/Core/macros/default.fwc > default.fwc
cvs co -p Fireworks/Core/macros/ispy.fwc > ispy.fwc
cvs co -p Fireworks/Core/macros/pflow.fwc > pflow.fwc
cvs co -p Fireworks/Core/macros/hflego.fwc > hflego.fwc

cd ${tard}/src
cvs co Fireworks/Core/icons
cvs co Fireworks/Core/data

