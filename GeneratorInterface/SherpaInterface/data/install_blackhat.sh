#!/bin/bash
#
#  file:        install_BlackHat.sh
#  description: BASH script for the installation of the BlackHat package,
#               can be used standalone or called from other scripts
#
#  author:      Markus Merschmeyer, Sebastian Thüer, RWTH Aachen University
#  date:        2014/02/22
#  version:     1.1
#

print_help() {
    echo "" && \
    echo "install_blackhat version 1.0" && echo && \
    echo "options: -v  version    define BlackHat version ( "${BHVER}" )" && \
    echo "         -d  path       define BlackHat installation directory" && \
    echo "                         -> ( "${IDIR}" )" && \
    echo "         -p  path       apply official BLACKHAT patches ( "${PATCHES}" )" && \
    echo "                         from this path ( "${PDIR}" )" && \
    echo "         -Q  path       qd installation directory ( "${QDDIR}" )" && \
    echo "         -W  location   (web)location of BlackHat tarball ( "${BHWEBLOCATION}" )" && \
    echo "         -S  filename   file name of BlackHat tarball ( "${BHFILE}" )" && \
    echo "         -C  level      cleaning level of SHERPA installation ( "${LVLCLEAN}" )" && \
    echo "                         -> 0: nothing, 1: +objects (make clean)" && \
    echo "         -D             debug flag, compile with '-g' option ( "${FLGDEBUG}" )" && \
    echo "         -X             create XML file for tool override in CMSSW ( "${FLGXMLFL}" )" && \
    echo "         -Z             use multiple CPU cores if available ( "${FLGMCORE}" )" && \
    echo "         -K             keep BlackHat source code tree after installation ( "${FGLKEEPT}" )" && \
    echo "         -h             display this help and exit" && echo
}


# save current path
HDIR=`pwd`


# get absolute path of this script
SCRIPT=$(readlink -f $0)
#echo "XXX: "$SCRIPT
SCRIPTPATH=`dirname $SCRIPT`
#echo "YYY: "$SCRIPTPATH


# dummy setup (if all options are missing)
IDIR="/tmp"                # installation directory
BHVER="0.9.9"              # BlackHat version to be installed
BHWEBLOCATION=""           # (web)location of BlackHat tarball
BHFILE=""                  # file name of BlackHat tarball
QDDIR=""                   # qd installation directory
LVLCLEAN=0                 # cleaning level (0-2)
FLGDEBUG="FALSE"           # debug flag for compilation
FLGXMLFL="FALSE"           # create XML tool definition file for SCRAM?
FLGKEEPT="FALSE"           # keep the source code tree?
FLGMCORE="FALSE"           # use multiple cores for compilation
PATCHES="FALSE"            # apply SHERPA patches
PDIR="./"                  # path containing patches



# get & evaluate options
while getopts :v:d:p:W:S:C:Q:DXZKh OPT
do
  case $OPT in
  v) BHVER=$OPTARG ;;
  d) IDIR=$OPTARG ;;
  Q) QDDIR=$OPTARG ;;
  W) BHWEBLOCATION=$OPTARG ;;
  S) BHFILE=$OPTARG ;;
  C) LVLCLEAN=$OPTARG ;;
  D) FLGDEBUG=TRUE ;;
  X) FLGXMLFL=TRUE ;;
  Z) FLGMCORE=TRUE ;;
  K) FLGKEEPT=TRUE ;;
  p) PATCHES=TRUE && PDIR=$OPTARG ;;  
  h) print_help && exit 0 ;;
  \?)
    shift `expr $OPTIND - 1`
    if [ "$1" = "--help" ]; then print_help && exit 0;
    else 
      echo -n "install_blackhat: error: unrecognized option "
      if [ $OPTARG != "-" ]; then echo "'-$OPTARG'. try '-h'"
      else echo "'$1'. try '-h'"
      fi
      print_help && exit 1
    fi
#    shift 1
#    OPTIND=1
  esac
done


# set up file names
MSI=$HDIR                            # main installation directory
MSI=$SCRIPTPATH
bhpatchfile="blackhat_patches_"${BHVER}".tgz" # official patches for current BLACKHAT version

# set BlackHat download location
if [ "$BHWEBLOCATION" = "" ]; then
  BHWEBLOCATION="http://www.hepforge.org/archive/blackhat"
  FLOC=" "
else
  if [ -e ${BHWEBLOCATION} ]; then   # is the location a local subdirectory?
    if [ -d ${BHWEBLOCATION} ]; then
      cd ${BHWEBLOCATION}; BHWEBLOCATION=`pwd`; cd ${HDIR}
    fi
  fi
  FLOC=" -W "${BHWEBLOCATION}
fi
if [ "$BHFILE" = "" ]; then
  BHFILE="blackhat-"${BHVER}".tar.gz"
fi


# make BlackHat version a global variable
export BHVER=${BHVER}
export QDVER=${QDVER}
# always use absolute path name...
cd ${PDIR}; PDIR=`pwd`; cd ${HDIR}
cd ${IDIR}; IDIR=`pwd`

echo " BlackHat installation: "
echo "  -> BlackHat version: '"${BHVER}"'"
echo "  -> installation directory: '"${IDIR}"'"
echo "  -> BlackHat location: '"${BHWEBLOCATION}"'"
echo "  -> BlackHat file name: '"${BHFILE}"'"
echo "  -> QD inst. dir.: '"${QDDIR}"'"
echo "  -> cleaning level: '"${LVLCLEAN}"'"
echo "  -> debugging mode: '"${FLGDEBUG}"'"
echo "  -> CMSSW override: '"${FLGXMLFL}"'"
echo "  -> keep sources:   '"${FLGKEEPT}"'"
echo "  -> use multiple CPU cores: '"${FLGMCORE}"'"
echo "  -> BlackHat patches: '"${PATCHES}"' in '"${PDIR}"'"


# set path to local BlackHat installation
export BHDIR=${IDIR}"/blackhat-"${BHVER}
export BHIDIR=${IDIR}"/BH_"${BHVER}



# add compiler & linker flags
echo "CXX      (old):  "$CXX
echo "CXXFLAGS (old):  "$CXXFLAGS
echo "LDFLAGS  (old):  "$LDFLAGS
##MM FIXME
#  export CXX=""
#  export CXXFLAGS=""
#  export LDFLAGS=""
##MM FIXME
if [ "$FLGDEBUG" = "TRUE" ]; then
  CFDEBUG="-g"
  export CXXFLAGS=${CXXFLAGS}" "${CFDEBUG}
fi
echo "CXX      (new):  "$CXX
echo "CXXFLAGS (new):  "$CXXFLAGS
echo "LDFLAGS  (new):  "$LDFLAGS

# add compiler & linker flags
COPTS="CXXFLAGS=-Wno-deprecated"
#intel compiler#COPTS=${COPTS}" CXXFLAGS=-wd803"
##COPTS="CXXFLAGS=-Wno-deprecated"
MOPTS=""
POPTS=""
BHCFLAGS="--with-QDpath="${QDDIR}
if [ "$FLGMCORE" = "TRUE" ]; then
    nprc=`cat /proc/cpuinfo | grep  -c processor`
    let nprc=$nprc
    if [ $nprc -gt 1 ]; then
      echo " <I> multiple CPU cores detected: "$nprc
      POPTS=" -j"$nprc" "
    fi
fi



# download, extract compile/install BlackHat
cd ${IDIR}
if [ ! -d ${BHIDIR} ]; then
  if [ `echo ${BHWEBLOCATION} | grep -c "http:"` -gt 0 ]; then
    echo " -> downloading BlackHat "${BHVER}" from "${BHWEBLOCATION}/${BHFILE}
    wget ${BHWEBLOCATION}/${BHFILE}
  elif [ `echo ${BHWEBLOCATION} | grep -c "srm:"` -gt 0 ]; then
    echo " -> srm-copying BlackHat "${BHVER}" from "${BHWEBLOCATION}/${BHFILE}
    srmcp ${BHWEBLOCATION}/${BHFILE} file:////${BHFILE}
  else
    echo " -> copying BlackHat "${BHVER}" from "${BHWEBLOCATION}/${BHFILE}
    cp ${BHWEBLOCATION}/${BHFILE} ./
  fi
  tar -xzf ${BHFILE}
  if [ ! "$FLGKEEPT" = "TRUE" ]; then
    rm ${BHFILE}
  fi
  cd ${BHDIR}
#  if [ ! -e configure ]; then
#    ./bootstrap
#  fi

### CHANGES 
# apply the necessary patches
  
  if [ "$PATCHES" = "TRUE" ]; then
	echo " <I> applying patches to BLACKHAT..."
	if [ -e ${PDIR}/${bhpatchfile} ]; then
		pfilelist=`tar -xzvf ${PDIR}/${bhpatchfile}`
		for pfile in `echo ${pfilelist}`; do
			echo "  -> applying patch: "${pfile}
			patch -p1 < ${pfile}
			echo " <I> (patches) removing file "${pfile}
			rm ${pfile}
		done
	else
		echo " <W> file "${PDIR}/${bhpatchfile}" does not exist,"
		echo " <W>  cannot apply Sherpa patches"
	fi
  fi
#### CHANGES



  echo " -> configuring BlackHat with options "${COPTS} && \
  ./configure --prefix=${BHIDIR} ${BHCFLAGS} ${COPTS} && \
  echo " -> making BlackHat with options "${POPTS} ${MOPTS} && \
  make ${POPTS} ${MOPTS} && \
  echo " -> installing BlackHat with options "${MOPTS} && \
  make install ${MOPTS}
  if [ ${LVLCLEAN} -gt 0 ]; then 
    echo " -> cleaning up BlackHat installation, level: "${LVLCLEAN}" ..."
    if [ ${LVLCLEAN} -ge 1 ]; then  # normal cleanup (objects)
      make clean
    fi
  fi
  cd ${HDIR}
  if [ "$FLGKEEPT" = "TRUE" ]; then
    echo "-> keeping source code..."
  else
    rm -rf ${BHDIR}
  fi
else
  echo " <W> path exists => using already installed BlackHat"
fi
export BHDIR=${BHIDIR}
cd ${HDIR}


# create XML file fro SCRAM
if [ "${FLGXMLFL}" = "TRUE" ]; then
  xmlfile="blackhat_"${BHVER}".xml"
  echo " <I>"
  echo " <I> creating BlackHat tool definition XML file"
  if [ -e ${xmlfile} ]; then rm ${xmlfile}; fi; touch ${xmlfile}
  echo "  <tool name=\"BlackHat\" version=\""${BHVER}"\">" >> ${xmlfile}
  tmppath=`find ${BHDIR} -type f -name libBH.so\*`
  tmpcnt=`echo ${tmppath} | grep -o "/" | grep -c "/"`
  tmppath=`echo ${tmppath} | cut -f 1-${tmpcnt} -d "/"`
  for LIB in `cd ${tmppath}; ls *.so | cut -f 1 -d "." | sed -e 's/lib//'; cd ${HDIR}`; do
    echo "    <lib name=\""${LIB}"\"/>" >> ${xmlfile}
  done
  echo "    <client>" >> ${xmlfile}
  echo "      <Environment name=\"BLACKHAT_BASE\" value=\""${BHDIR}"\"/>" >> ${xmlfile}
  echo "      <Environment name=\"LIBDIR\" default=\"\$BLACKHAT_BASE/lib/blackhat\"/>" >> ${xmlfile}
  echo "      <Environment name=\"INCLUDE\" default=\"\$BLACKHAT_BASE/include/blackhat\"/>" >> ${xmlfile}
  echo "    </client>" >> ${xmlfile}
#  echo "    <runtime name=\"CMSSW_FWLITE_INCLUDE_PATH\" value=\"\$BLACKHAT_BASE/include\" type=\"path\"/>" >> ${xmlfile}
#  echo "    <use name=\"CLHEP\"/>" >> ${xmlfile}
  echo "  </tool>" >> ${xmlfile}
  if [ ! "$PWD" = "${HDIR}" ]; then
    mv ${xmlfile} ${HDIR}/
  fi

  if [ ! "$CMSSW_BASE" = "" ]; then
    cd $CMSSW_BASE
    tmpbh=`scramv1 tool info blackhat | grep "BLACKHAT_BASE" | cut -f2 -d"="`
    tmpxml=`find $CMSSW_BASE/config -type f -name blackhat.xml -printf %h`
    echo " <I>"
    echo " <I> BlackHat version currently being used: "${tmpbh}
    echo " <I> ...defined in "${tmpxml}
    cd ${tmpxml}; tmpxml=$PWD; cd ${HDIR}
    echo " <I>"
    echo " <I> If you want to override this version with the freshly produced "${xmlfile}","
    echo " <I> ...please type the following commands:"
    echo " <I>"
    echo "       cd $CMSSW_BASE"
    echo "       scramv1 tool remove blackhat"
    echo "       cp ${HDIR}/${xmlfile} ${tmpxml}/"
    echo "       scramv1 setup blackhat"
    echo "       cd -"
    echo " <I>"
  fi

fi


echo " -> BlackHat installation directory is: "
echo "  "${BHIDIR}
