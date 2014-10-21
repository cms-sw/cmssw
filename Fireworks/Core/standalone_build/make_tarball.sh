#!/bin/bash

getExternals()
{
   mkdir  ${tard}/external

   # external libraries
    extdl=${tard}/external/lib
    mkdir $extdl
    extt=/tmp/cmsswExt
    ext=`dirname ${CMSSW_DATA_PATH}`/external
    ls -l $CMSSW_RELEASE_BASE/external/$SCRAM_ARCH/lib/* > $extt
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
        ever=`grep $i $extt |  perl -ne 'if ($_ =~ /$ENV{i}\/(.*)\/lib\/(.*)/ ) {print "$1\n"; last;}'`
        echo "copy $i $ever"
        if [ -z "$ever" ]; then
            echo "!!!!!!!! can't get externals for $i"
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
     
}
#----------------------------------------------------------------

getCmssw()
{
    echo "=========================================================="
    echo "=========================================================="
    echo "get CMS libs"

    mkdir -p ${tard}/lib
    fwl="/tmp/fwlite_build_set.file"
    $dwnCmd $fwl https://raw.githubusercontent.com/cms-sw/cmsdist/IB/CMSSW_7_0_X/stable/fwlite_build_set.file
    
    # remove package without libs
    perl -i -ne 'print unless /Fireworks\/Macros/' $fwl
    perl -i -ne 'print unless /FWCore\/PythonUtilities/' $fwl
    perl -i -ne 'print unless /DataFormats\/MuonData/' $fwl
    perl -i -ne 'print unless /Utilities\/ReleaseScripts/' $fwl

    cn=${tard}/lib/.edmplugincache;
    fwpl=`cat $fwl |  perl -ne 'if( ~/(.+)\/(.+)$/){print "$1$2 ";}'`
    echo "get list from $fwpl"
    for i in $fwpl
    do
	cp -f $CMSSW_RELEASE_BASE/lib/$SCRAM_ARCH/*${i}* $tard/lib
	grep $i  $CMSSW_RELEASE_BASE/lib/$SCRAM_ARCH/.edmplugincache  >> $cn

    done;
    
    echo "getting libs from $CMSSW_BASE/lib/$SCRAM_ARCH"
    cp -f $CMSSW_BASE/lib/$SCRAM_ARCH/* ${tard}/lib/

    sort -u  $cn -o  $cn
}

#----------------------------------------------------------------

getSources()
{
   echo "=========================================================="
   echo "=========================================================="
   echo "getting Fireworks info/config files."
   # binary 
   mkdir ${tard}/libexec
   cp $CMSSW_RELEASE_BASE/bin/$SCRAM_ARCH/cmsShow.exe ${tard}/libexec
   cp $CMSSW_RELEASE_BASE/bin/$SCRAM_ARCH/cmsShowSendReport ${tard}/libexec

   if [ -e $CMSSW_BASE/bin/$SCRAM_ARCH/cmsShow.exe ]; then 
      cp -f $CMSSW_BASE/bin/$SCRAM_ARCH/cmsShow.exe ${tard}/libexec
   fi
   if [ -e $CMSSW_BASE/bin/$SCRAM_ARCH/cmsShowSendReport ]; then 
      cp -f $CMSSW_BASE/bin/$SCRAM_ARCH/cmsShowSendReport ${tard}/libexec
   fi
   
   # src
   srcDir="${tard}/src/Fireworks/Core"
   mkdir -p $srcDir
   cp -a  $CMSSW_BASE/src/Fireworks/Core/macros $srcDir
   cp -a  $CMSSW_BASE/src/Fireworks/Core/icons $srcDir
   cp -a  $CMSSW_BASE/src/Fireworks/Core/data $srcDir
   cp -a  $CMSSW_BASE/src/Fireworks/Core/scripts $srcDir
   
   # version info
   cv=$tversion;
   if [ -z $cv ]; then
      cv=`perl -e 'if ($ENV{CMSSW_VERSION} =~ /CMSSW_(\d+)_(\d+)_/) {print "${1}.${2}";}'`
   fi
   echo $cv > $srcDir/data/version.txt
   echo "DataFormats: $CMSSW_VERSION" >> $srcDir/data/version.txt
   # cat $srcDir/data/version.txt
   cp -a $CMSSW_RELEASE_BASE/src/Fireworks/Core/scripts $srcDir

   # link to config files
   cd  $tard
   ln -s  src/Fireworks/Core/macros/default.fwc .
   ln -s  src/Fireworks/Core/macros/ispy.fwc .
   ln -s  src/Fireworks/Core/macros/pflow.fwc .
   ln -s  src/Fireworks/Core/macros/hfLego.fwc .
   ln -s  src/Fireworks/Core/macros/simGeo.fwc .
   ln -s  src/Fireworks/Core/macros/overlaps.fwc .
   
   ln -s  src/Fireworks/Core/scripts/cmsShow .   
}

#----------------------------------------------------------------

getDataFiles()
{
   echo "=========================================================="
   echo "=========================================================="
   echo "get data files"
   # sample files
   cd ${tard}
   name=`perl -e '($ver, $a, $b, $c) = split('_', $ENV{CMSSW_VERSION}); print  "data", $a, $b, ".root"  '`
   $dwnCmd data.root  http://amraktad.web.cern.ch/amraktad/mail/scratch0/data/$name

   mc_name=`perl -e '($ver, $a, $b, $c) = split('_', $ENV{CMSSW_VERSION}); print  "mc", $a, $b, ".root"  '`
   $dwnCmd mc.root http://amraktad.web.cern.ch/amraktad/mail/scratch0/data/$mc_name

   #geometry files
   cp $CMSSW_RELEASE_BASE/external/$SCRAM_ARCH/data/Fireworks/Geometry/data/cmsSimGeom-* ${tard}
   cp $CMSSW_RELEASE_BASE/external/$SCRAM_ARCH/data/Fireworks/Geometry/data/cmsGeom* ${tard}
}

#----------------------------------------------------------------

makeTar()
{
   bdir=`dirname $tard`
   cd $bdir
   tname=`basename $tard`
   if [ `uname` = "Darwin" ]; then
       echo "Packing tarball ${tname}.mac.tar.gz"
       tar -czf ${tname}.mac.tar.gz $tname
   else
       echo "Packing tarball ${tname}.linux.tar.gz"
       tar -czf ${tname}.linux.tar.gz $tname
   fi
}


#----------------------------------------------------------------
#----------------------------------------------------------------
#----------------------------------------------------------------

usage() { echo "Usage: $0  --tar --version=<version> --dir=<destDir> --verbose --force" ; exit;}

for i in "$@"
do
case $i in
    --dir=*)
    tard="${i#*=}"
    echo "Destination directory  == [$tard]"
    shift;
    ;;
    --version=*)
    tversion="${i#*=}"
    echo "Tarball version  == $tversion"
    shift;
    ;;
    --tar)
    doTar=YES
    echo "Do tar&zip after extraction"
    shift;
    ;;
    --verbose)
    set -x
    shift;
    ;;
    --force)
    doForce=1;
    shift;
    ;;
    *)
    usage
    ;;
esac
done

export tard
tard=`perl -e '{$a = $ENV{tard}; $a =~ s/^~/$ENV{HOME}/;$a =~ s/^\./$ENV{PWD}/; print "$a"; }'`


if [ -z $tard ]; then
echo "Destination directory not specified"
usage
fi

echo -e "Start packaging .... \n"



if [ -z $doForce ] && [ -e $tard ] ; then
   echo "Destination directory  [$tard] already exists. Use --force option."
   exit 1;
fi

mkdir $tard

dwnCmd="wget --no-check-certificate -O "
if [ `uname` = "Darwin" ]; then
    dwnCmd="curl --insecure -o "
fi

origd=$PWD
getExternals
getCmssw


getSources
getDataFiles
echo $tard
if [ -n "$doTar" ] ; then
   makeTar
fi

