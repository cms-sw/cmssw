#!/bin/bash

   dwnCmd="wget --no-check-certificate"
   if [ `uname` = "Darwin" ]; then
      # compatibility problem on 10.5
      dwnCmd="curl -k -O"
   fi

getExternals()
{
    mkdir  ${tard}/external

   # external libraries
    extdl=${tard}/external/lib
    mkdir $extdl
    
    ext=`dirname ${CMSSW_DATA_PATH}`/external
    ls -l $CMSSW_RELEASE_BASE/external/$SCRAM_ARCH/lib/* > $HOME/extlist

    echo "=========================================================="
    echo "=========================================================="

    gccd=${tard}/external

	export gcmd=`which gcc`
	gv=`perl -e ' if ($ENV{gcmd} =~ /\/gcc\/(.*)\/bin\/gcc/) { print $1;}'`
	printf "gcc version $gv"
	if [ -z "$gv" ]; then
            echo "can't get gcc version"
            exit;
	fi
	echo "Copy gcc from  $ext/gcc/${gv}/ to ${gccd}"
	cp -a $ext/gcc/${gv}/ ${gccd}/gcc
        if [ `uname` = "Darwin" ]; then
           echo "Renaming gcc lib directory to lib64."
           mv ${gccd}/gcc/lib ${gccd}/gcc/lib64
        fi
    
    echo "=========================================================="
    echo "=========================================================="


    echo "Copying external libraries from $ext to $extdl."
   # cp -a $ext/*/*/lib/*  ${tard}/external/lib
    for i in boost bz2lib castor clhep dcap db4 dcap \
        expat fftw3 gdbm gsl hepmc\
   	libjpg libpng libtiff libungif \
   	openssl pcre \
   	sigcpp  xrootd zlib xz freetype
    do
        export i;
        ever=`grep $i $HOME/extlist |  perl -ne 'if ($_ =~ /$ENV{i}\/(.*)\/lib\/(.*)/ ) {print "$1\n"; last;}'`
        echo "copy $i $ever"
        if [ -z "$ever" ]; then
            echo "warning can't get externals for $i"
	fi
        # echo "cp -a $ext/$i/$ever/lib/* === ${extdl}"
        cp -a $ext/$i/$ever/lib/* ${extdl}
    done


    echo "=========================================================="
    echo "=========================================================="
   # can be linked or installed at $ROOTSYS   
    ROOTSYS=`echo $ROOTSYS |  sed 's/\/$//'` # remove '/' character  at end of string, becuse it invalidates symblic link interpretation
    origr=$ROOTSYS
    if [ -L ${ROOTSYS} ]; then
	b=`dirname ${ROOTSYS}`   
	if [ `uname` = "Linux" ]; then
            origr=`readlink -f ${ROOTSYS}`
	else
            origr=`readlink ${ROOTSYS}`
	fi
    fi
    
    echo "copy root from $origr to ${tard}/external/root"
   # pushd $PWD
   # cd $ROOTSYS
   # ROOTSYS=${tard}/external/root make install
   # popd
    cp -a $origr  ${tard}/external/root
   
 
   
   rm $HOME/extlist;
   
}
   
#----------------------------------------------------------------

getCmssw()
{
   echo "=========================================================="
   echo "=========================================================="

   # cms libraries
   mkdir -p ${tard}/lib
   
   $dwnCmd  https://raw.github.com/cms-sw/cmsdist/IB/CMSSW_7_0_X/stable/fwlite_build_set.file

   while read p; do
       echo "get libs from $p"
       x=`dirname $p`
       y=`basename $p`
       cp -a $CMSSW_RELEASE_BASE/lib/*/*${x}*${y}* ${tard}/lib/
       #if [ -f "$file" ]
	   cp -f $CMSSW_BASE/lib/*/*${x}*${y}* ${tard}/lib/ 2>/dev/null
       #fi
    done < fwlite_build_set.file

   # plugins cache file
      echo "get $CMSSW_RELEASE_BASE/lib/*/.edmplugincache > ${tard}/lib/.edmplugincache"
      touch ${tard}/lib/.edmplugincache
      cat  $CMSSW_RELEASE_BASE/lib/*/.edmplugincache > /tmp/.edmplugincache
      cat  $CMSSW_BASE/lib/*/.edmplugincache >> /tmp/.edmplugincache
      cat /tmp/.edmplugincache | sort -u >  ${tard}/lib/.edmplugincache
}

#----------------------------------------------------------------

getSources()
{
   echo "=========================================================="
   echo "getting sources."
   # binary 
   mkdir ${tard}/libexec
   cp $CMSSW_RELEASE_BASE/bin/*/cmsShow.exe ${tard}/libexec
   cp $CMSSW_RELEASE_BASE/bin/*/cmsShowSendReport ${tard}/libexec
   
   # src
   srcDir="${tard}/src/Fireworks/Core"
   mkdir -p $srcDir
   cp -a  $CMSSW_RELEASE_BASE/src/Fireworks/Core/macros $srcDir
   cp -a  $CMSSW_RELEASE_BASE/src/Fireworks/Core/icons $srcDir
   cp -a  $CMSSW_RELEASE_BASE/src/Fireworks/Core/data $srcDir

   cv=`perl -e 'if ($ENV{CMSSW_VERSION} =~ /CMSSW_(\d+)_(\d+)_/) {print "${1}.${2}";}'`
   echo $cv > $srcDir/data/version.txt
   echo "DataFormats: $CMSSW_VERSION" >> $srcDir/data/version.txt
   cat $srcDir/data/version.txt
   cp -a  $CMSSW_RELEASE_BASE/src/Fireworks/Core/scripts $srcDir
   
   cd  $tard
   ln -s  src/Fireworks/Core/macros/default.fwc .
   ln -s  src/Fireworks/Core/macros/ispy.fwc .
   ln -s  src/Fireworks/Core/macros/pflow.fwc .
   ln -s  src/Fireworks/Core/macros/hfLego.fwc .
   ln -s  src/Fireworks/Core/macros/simGeo.fwc .
   ln -s  src/Fireworks/Core/macros/overlaps.fwc .
   
   ln -s  src/Fireworks/Core/scripts/cmsShow .
   
   cp  $CMSSW_DATA_PATH/data-Fireworks-Geometry/4-cms/Fireworks/Geometry/data/* $tard
}

#----------------------------------------------------------------

getDataFiles()
{
   # sample files
   cd $tard
   cd ${tard}
   name=`perl -e '($ver, $a, $b, $c) = split('_', $ENV{CMSSW_VERSION}); print  "data", $a, $b, ".root"  '`
   $dwnCmd http://amraktad.web.cern.ch/amraktad/mail/scratch0/data/$name
   mv $name data.root

   mc_name=`perl -e '($ver, $a, $b, $c) = split('_', $ENV{CMSSW_VERSION}); print  "mc", $a, $b, ".root"  '`
   $dwnCmd http://amraktad.web.cern.ch/amraktad/mail/scratch0/data/$mc_name
   mv $mc_name mc.root

   $dwnCmd http://amraktad.web.cern.ch/amraktad/mail/scratch0/data/cmsSimGeom-14.root
   $dwnCmd http://amraktad.web.cern.ch/amraktad/mail/scratch0/data/cmsGeom10.root
}

#----------------------------------------------------------------

makeTar()
{
   cd $origd
   bdir=`basename $tard`
   if [ `uname` = "Darwin" ]; then
       echo "Packing tarball ${bdir}.mac.tar.gz"
       tar -czf ${bdir}.mac.tar.gz $bdir
   else
       echo "Packing tarball ${bdir}.linux.tar.gz"
       tar -czf ${bdir}.linux.tar.gz $bdir
   fi
}


#----------------------------------------------------------------
#----------------------------------------------------------------
#----------------------------------------------------------------

if [ $# -lt 1 ]; then
  echo "Usage: $0  [-t] [-v] destination_dir "
  exit 1
fi


while [ $# -gt 0 ]; do
   case "$1" in
    -t)  doTar=on;;
    -v)  verbose=on;
   esac
   tard=$1
   shift
done


if [[ "$tard" = /* ]]
then
   : # Absolute path
else
   : tard=$PWD/$tard
fi

echo "dirpath = $tard"

if [ "X$verbose" = Xon ] ; then
   set -xv
fi

if [ -f $tard ]; then
   echo "dir exist alreday"
   exit
fi

mkdir $tard

origd=$PWD
getExternals
getCmssw
getSources
getDataFiles
echo $tard
if [ "X$doTar" = Xon ] ; then
   makeTar
fi

