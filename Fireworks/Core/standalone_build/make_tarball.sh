#!/bin/bash

getExternals()
{
    mkdir  ${tard}/external

    ext=$CMS_PATH/$SCRAM_ARCH/external # this used to be CMSSW_DATA_PATH
    echo "=========================================================="
    echo "=========================================================="

    cd $CMSSW_BASE
	echo "Copy gcc subdirs"
    scram tool tag gcc-ccompiler GCC_CCOMPILER_BASE
    gccd_src=`scram tool tag gcc-ccompiler GCC_CCOMPILER_BASE`
set -x
    gccd_target=${tard}/external/gcc
    mkdir $gccd_target
	cp -a $gccd_src/include $gccd_target
	cp -a $gccd_src/bin $gccd_target

    if [ `uname` = "Darwin" ]; then
   	   cp -a $gccd_src/lib $gccd_target/lib64
    else
       cp -a $gccd_src/lib64 $gccd_target/lib64
    fi
    exit
    echo "=========================================================="
    echo "=========================================================="

    echo "Copying external libraries from $ext to $extdl."

   # external libraries
    extdl=${tard}/external/lib
    mkdir $extdl
    extt=/tmp/cmsswExt
    ls -l $CMSSW_RELEASE_BASE/external/$SCRAM_ARCH/lib/* > $extt
   # cp -a $ext/*/*/lib/*  ${tard}/external/lib
    for i in boost bz2lib castor clhep dcap db4 dcap \
        expat fftw3 gdbm gsl hepmc\
   	libjpg libpng libtiff libungif \
   	openssl pcre \
   	sigcpp  xrootd zlib xz freetype tbb
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
    echo "Copying external headers."

    mkdir ${tard}/external/var-inc
    for i in CLHEP HepMC boost sigcpp; do
      # scram tool info $i | grep INCL | head -1
       edir=`scram tool info $i | grep INCL | head -1| perl -ne 'if ($_ =~/\=(.*)$/) {print "$1\n"}'`
       if [ -n $edir ]; then
          cp -r $edir/* ${tard}/external/var-inc
       fi
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
getCmsSources()
{
    echo "get list from $fwpl"
    for i in `cat $fwl` $extra_list
    do
        # run-away headers
        if [ -f $CMSSW_RELEASE_BASE/src/${i} ]; then
           mkdir -p  $tard/src/`dirname $i`
           cp $CMSSW_RELEASE_BASE/src/$i $tard/src/$i
        fi


        mkdir -p $tard/src/$i
        mkdir -p $tard/src/$i/src

        if [ -e  $CMSSW_BASE/src/$i ]; then
	   cp -rf $CMSSW_BASE/src/${i}/interface $tard/src/$i
	   cp -rf $CMSSW_BASE/src/${i}/src/*.h $tard/src/$i/src
        else
	   cp -rf $CMSSW_RELEASE_BASE/src/${i}/interface $tard/src/$i
	   cp -rf $CMSSW_RELEASE_BASE/src/${i}/src/*.h $tard/src/$i/src
        fi
    done;
}

#----------------------------------------------------------------

getFireworksSources()
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
   ln -s  src/Fireworks/Core/macros/aod.fwc .
   ln -s  src/Fireworks/Core/macros/ispy.fwc .
   ln -s  src/Fireworks/Core/macros/pflow.fwc .
   ln -s  src/Fireworks/Core/macros/miniaod.fwc .
   ln -s  src/Fireworks/Core/macros/simGeo.fwc .
   ln -s  src/Fireworks/Core/macros/overlaps.fwc .
   
   ln -s  src/Fireworks/Core/scripts/cmsShow .   
}

#----------------------------------------------------------------

getCmsLibs()
{
    echo "=========================================================="
    echo "=========================================================="
    echo "get CMS libs"

    libext=".so";
    if [ `uname` = "Darwin" ]; then
      libext=".dylib";
    fi

    cat $fwl | grep -v '\.h$' >  ${fwl}tmp
    # remove package without libs
    perl -i -ne 'print unless /Fireworks\/Macros/'         ${fwl}tmp
    perl -i -ne 'print unless /FWCore\/PythonUtilities/'   ${fwl}tmp
    perl -i -ne 'print unless /DataFormats\/MuonData/'     ${fwl}tmp
    perl -i -ne 'print unless /Utilities\/ReleaseScripts/' ${fwl}tmp


    libl=`cat ${fwl}tmp |  perl -ne 'if( ~/(.+)\/(.+)$/){print "$1$2 ";}'`
    libl_extra=`echo $extra_list | perl -pe '{ s/\///og;}'`

    echo "get FWLite libraries"

    cn=${tard}/lib/.edmplugincache;
    for i in $libl $libl_extra; do
       if [ -e  $CMSSW_BASE/lib/$SCRAM_ARCH/lib${i}.$libext ]; then
	      cp -f $CMSSW_BASE/lib/$SCRAM_ARCH/*${i}* $tard/lib
       else
	      cp -f $CMSSW_RELEASE_BASE/lib/$SCRAM_ARCH/*${i}* $tard/lib
       fi
	   grep $i  $CMSSW_RELEASE_BASE/lib/$SCRAM_ARCH/.edmplugincache  >> $cn
	   grep $i  $CMSSW_BASE/lib/$SCRAM_ARCH/.edmplugincache  >> $cn
    done

    echo "getting libs from $CMSSW_BASE/lib/$SCRAM_ARCH"
    cp -f $CMSSW_BASE/lib/$SCRAM_ARCH/* ${tard}/lib/

    sort -u  $cn -o  $cn
    
    rm ${fwl}tmp
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
   $dwnCmd data.root  http://amraktad.web.cern.ch/amraktad/scratch0/data/$name

 #  mc_name=`perl -e '($ver, $a, $b, $c) = split('_', $ENV{CMSSW_VERSION}); print  "mc", $a, $b, ".root"  '`
 #  $dwnCmd mc.root http://amraktad.web.cern.ch/amraktad/mail/scratch0/data/$mc_name

   #geometry files
   cp $CMSSW_RELEASE_BASE/external/$SCRAM_ARCH/data/Fireworks/Geometry/data/cmsSimGeom-* ${tard}
   cp $CMSSW_RELEASE_BASE/external/$SCRAM_ARCH/data/Fireworks/Geometry/data/cmsGeom10.root ${tard}
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
    --fwlite=*)
    fwlite_list="${i#*=}"
    shift;
    ;;
    *)
echo "usage [$i] ----"
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

if [ -z fwlite_list ]; then
   fwl="/tmp/fwlite_build_set.file"
   $dwnCmd $fwl https://raw.githubusercontent.com/cms-sw/cmsdist/IB/CMSSW_7_3_X/stable/fwlite_build_set.file
else
   fwl=$fwlite_list
fi
extra_list="/CondFormats/Serialization /Geometry/CommonDetUnit /DataFormats/MuonSeed"


getExternals
getCmsSources
getFireworksSources

mkdir -p ${tard}/lib
getCmsLibs


getDataFiles
echo $tard
if [ -n "$doTar" ] ; then
   makeTar
fi

