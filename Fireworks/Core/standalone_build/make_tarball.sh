#!/bin/bash
set -xv

if [ $# -lt 1 ]; then
  echo "Usage: $0  [tar-dir]"
  exit 1
fi

# TODO: add 'std' option so file name is automatically created from release name and date
tard=${PWD}/$1

# TODO: sample file location should be external parameter or downloaded from web
dataDir=${PWD}/data

#----------------------------------------------------------------
# root
# can be linked or installed at $ROOTSYS

mkdir  ${tard}/external
ROOTSYS=`echo $ROOTSYS |  sed 's/\/$//'` # remove '/' character  at end of string, becuse it invalidates symblic link interpretation
if [ -L ${ROOTSYS} ]; then
   b=${ROOTSYS}
   origr=`readlink ${ROOTSYS}`
   cd ${origr}
   ROOTSYS=${tard}/external/root make install
   mkdir -p ${tard}   ROOTSYS=${origr}
else
   cd $ROOTSYS   
   ROOTSYS=${tard}/external/root make install
fi

#----------------------------------------------------------------
# external libraries

mkdir ${tard}/external/lib
ext=`dirname ${CMSSW_DATA_PATH}`/external
cp -a $ext/*/*/lib/*  ${tard}/external/lib

if [ `uname` = "Darwin" ]; then
   # compatibility problem on 10.5
   rm  ${tard}/external/lib/libcrypto.*
fi

#----------------------------------------------------------------
# cmssw
   
mkdir -p ${tard}/lib

# cms libraries
cp $CMSSW_RELEASE_BASE/lib/*/* ${tard}/lib/
cp -f $CMSSW_BASE/lib/*/* ${tard}/lib/

# plugins cache file
touch ${tard}/lib/.edmplugincache
cat  $CMSSW_RELEASE_BASE/lib/*/.edmplugincache >> ${tard}/lib/.edmplugincache
cat  $CMSSW_BASE/lib/*/.edmplugincache >> ${tard}/lib/.edmplugincache

# binary 
cp $CMSSW_BASE/bin/*/cmsShow.exe ${tard}

# version file, icons, and configuration files
# TODO: should be in etc, but have to change Fireworks source code first

mkdir -p ${tard}/src
cd ${tard}/src
cvs co Fireworks/Core/macros
cvs co Fireworks/Core/icons
cvs co Fireworks/Core/data

#----------------------------------------------------------------

# sample files
dwnCmd="wget"
if [ `uname` = "Darwin" ]; then
   # compatibility problem on 10.5
   dwnCmd="curl -O"
fi
cd ${tard}
$dwnCmd http://amraktad.web.cern.ch/amraktad/mail/scratch0/data/data.root
$dwnCmd http://amraktad.web.cern.ch/amraktad/mail/scratch0/data/mc.root
$dwnCmd http://amraktad.web.cern.ch/amraktad/mail/scratch0/data/cmsGeom10.root
# temprary here till fwlite build is excepted
$dwnCmd http://amraktad.web.cern.ch/amraktad/mail/scratch0/data/cmsShow 
chmod a+x cmsShow

#----------------------------------------------------------------
# tar file

cd `dirname $tard`
tarname=`basename $tard`.`uname`.tar.gz
tar -czf ${tarname}  `basename ${tard}`

# rm -rf ${tard} 
# cp $tarname  /afs/cern.ch/cms/fireworks/standalone-build/
