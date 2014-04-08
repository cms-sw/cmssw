#!/bin/bash
#
#  file:        install_hepmc2.sh
#  description: BASH script for the installation of the HepMC2 package,
#               can be used standalone or called from other scripts
#
#  author:      Markus Merschmeyer, RWTH Aachen University
#  date:        2013/05/23
#  version:     3.0
#

print_help() {
    echo "" && \
    echo "install_hepmc2 version 3.0" && echo && \
    echo "options: -v  version    define HepMC2 version ( "${HEPMC2VER}" )" && \
    echo "         -d  path       define HepMC2 installation directory" && \
    echo "                         -> ( "${IDIR}" )" && \
    echo "         -W  location   (web)location of HepMC2 tarball ( "${HEPMC2WEBLOCATION}" )" && \
    echo "         -S  filename   file name of HepMC2 tarball ( "${HEPMC2FILE}" )" && \
    echo "         -C  level      cleaning level of SHERPA installation ( "${LVLCLEAN}" )" && \
    echo "                         -> 0: nothing, 1: +objects (make clean)" && \
    echo "         -D             debug flag, compile with '-g' option ( "${FLGDEBUG}" )" && \
    echo "         -X             create XML file for tool override in CMSSW ( "${FLGXMLFL}" )" && \
    echo "         -Z             use multiple CPU cores if available ( "${FLGMCORE}" )" && \
    echo "         -K             keep HEPMC2 source code tree after installation ( "${FLGKEEPT}" )" && \
    echo "         -h             display this help and exit" && echo
}


# save current path
HDIR=`pwd`


# dummy setup (if all options are missing)
IDIR="/tmp"                # installation directory
HEPMC2VER="2.03.06"        # HepMC2 version  to be installed
HEPMC2WEBLOCATION=""       # (web)location of HEPMC2 tarball
HEPMC2FILE=""              # file name of HEPMC2 tarball
LVLCLEAN=0                 # cleaning level (0-2)
FLGDEBUG="FALSE"           # debug flag for compilation
FLGXMLFL="FALSE"           # create XML tool definition file for SCRAM?
FLGKEEPT="FALSE"           # keep the source code tree?
FLGMCORE="FALSE"           # use multiple cores for compilation


# get & evaluate options
while getopts :v:d:W:S:C:DXZKh OPT
do
  case $OPT in
  v) HEPMC2VER=$OPTARG ;;
  d) IDIR=$OPTARG ;;
  W) HEPMC2WEBLOCATION=$OPTARG ;;
  S) HEPMC2FILE=$OPTARG ;;
  C) LVLCLEAN=$OPTARG ;;
  D) FLGDEBUG=TRUE ;;
  X) FLGXMLFL=TRUE ;;
  Z) FLGMCORE=TRUE ;;
  K) FLGKEEPT=TRUE ;;
  h) print_help && exit 0 ;;
  \?)
    shift `expr $OPTIND - 1`
    if [ "$1" = "--help" ]; then print_help && exit 0;
    else 
      echo -n "install_hepmc2: error: unrecognized option "
      if [ $OPTARG != "-" ]; then echo "'-$OPTARG'. try '-h'"
      else echo "'$1'. try '-h'"
      fi
      print_help && exit 1
    fi
#    shift 1
#    OPTIND=1
  esac
done


# set HEPMC2 download location
if [ "$HEPMC2WEBLOCATION" = "" ]; then
  HEPMC2WEBLOCATION="http://lcgapp.cern.ch/project/simu/HepMC/download"
else
  if [ -e ${HEPMC2WEBLOCATION} ]; then   # is the location a local subdirectory?
    if [ -d ${HEPMC2WEBLOCATION} ]; then
      cd ${HEPMC2WEBLOCATION}; HEPMC2WEBLOCATION=`pwd`; cd ${HDIR}
    fi
  fi
fi
if [ "$HEPMC2FILE" = "" ]; then
  HEPMC2FILE="HepMC-"${HEPMC2VER}".tar.gz"
fi


# make HEPMC2 version a global variable
export HEPMC2VER=${HEPMC2VER}
# always use absolute path name...
cd ${IDIR}; IDIR=`pwd`

echo " HepMC2 installation: "
echo "  -> HepMC2 version: '"${HEPMC2VER}"'"
echo "  -> installation directory: '"${IDIR}"'"
echo "  -> HepMC2 location: '"${HEPMC2WEBLOCATION}"'"
echo "  -> HepMC2 file name: '"${HEPMC2FILE}"'"
echo "  -> cleaning level: '"${LVLCLEAN}"'"
echo "  -> debugging mode: '"${FLGDEBUG}"'"
echo "  -> CMSSW override: '"${FLGXMLFL}"'"
echo "  -> keep sources:   '"${FLGKEEPT}"'"
echo "  -> use multiple CPU cores: '"${FLGMCORE}"'"

# analyze HEPMC2 version
va=`echo ${HEPMC2VER} | cut -f1 -d"."`
vb=`echo ${HEPMC2VER} | cut -f2 -d"."`
vc=`echo ${HEPMC2VER} | cut -f3 -d"."`
#echo "VA,VB,VC: "$va","$vb","$vc
if [ $va -ge 2 ] && [ $vb -ge 4 ]; then
# CMS conventions: https://twiki.cern.ch/twiki/bin/view/CMS/CMSConventions#CMS_units
  momflag="--with-momentum=GEV"
  lenflag="--with-length=CM"
#  momflag=""
else
  momflag=""
  lenflag=""
fi



# set path to local HEPMC2 installation
export HEPMC2DIR=${IDIR}"/HepMC-"${HEPMC2VER}
export HEPMC2IDIR=${IDIR}"/HEPMC_"${HEPMC2VER}


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
COPTS=""
MOPTS=""
POPTS=""
if [ "$FLGMCORE" = "TRUE" ]; then
    nprc=`cat /proc/cpuinfo | grep  -c processor`
    let nprc=$nprc
    if [ $nprc -gt 1 ]; then
      echo " <I> multiple CPU cores detected: "$nprc
      POPTS=" -j"$nprc" "
    fi
fi


# download, extract compile/install HEPMC2
cd ${IDIR}
#if [ ! -d ${HEPMC2DIR} ]; then
if [ ! -d ${HEPMC2IDIR} ]; then
  if [ `echo ${HEPMC2WEBLOCATION} | grep -c "http:"` -gt 0 ]; then
    echo " -> downloading HepMC2 "${HEPMC2VER}" from "${HEPMC2WEBLOCATION}/${HEPMC2FILE}
    wget ${HEPMC2WEBLOCATION}/${HEPMC2FILE}
  elif [ `echo ${HEPMC2WEBLOCATION} | grep -c "srm:"` -gt 0 ]; then
    echo " -> srm-copying HepMC2 "${HEPMC2VER}" from "${HEPMC2WEBLOCATION}/${HEPMC2FILE}
    srmcp ${HEPMC2WEBLOCATION}/${HEPMC2FILE} file:////${HEPMC2FILE}
  else
    echo " -> copying HepMC2 "${HEPMC2VER}" from "${HEPMC2WEBLOCATION}/${HEPMC2FILE}
    cp ${HEPMC2WEBLOCATION}/${HEPMC2FILE} ./
  fi
  tar -xzf ${HEPMC2FILE}
  if [ ! "$FLGKEEPT" = "TRUE" ]; then
    rm ${HEPMC2FILE}
  fi
  cd ${HEPMC2DIR}
  if [ ! -e configure ]; then
    ./bootstrap
  fi

  echo " -> configuring HepMC2 with options "${COPTS} && \
  ./configure --prefix=${HEPMC2IDIR} ${momflag} ${lenflag} ${COPTS} && \
  echo " -> making HepMC2 with options "${POPTS} ${MOPTS} && \
  make ${POPTS} ${MOPTS} && \
  echo " -> installing HepMC2 with options "${MOPTS} && \
  make install ${MOPTS}
  if [ ${LVLCLEAN} -gt 0 ]; then 
    echo " -> cleaning up HEPMC2 installation, level: "${LVLCLEAN}" ..."
    if [ ${LVLCLEAN} -ge 1 ]; then  # normal cleanup (objects)
      make clean
    fi
  fi
  cd ${HDIR}
  if [ "$FLGKEEPT" = "TRUE" ]; then
    echo "-> keeping source code..."
  else
    rm -rf ${HEPMC2DIR}
  fi
else
  echo " <W> path exists => using already installed HepMC2"
fi
export HEPMC2DIR=${HEPMC2IDIR}
cd ${HDIR}


# create XML file fro SCRAM
if [ "${FLGXMLFL}" = "TRUE" ]; then
#  xmlfile=hepmc.xml
  xmlfile="hepmc_"${HEPMC2VER}".xml"
  echo " <I>"
  echo " <I> creating HepMC tool definition XML file"
  if [ -e ${xmlfile} ]; then rm ${xmlfile}; fi; touch ${xmlfile}
  echo "  <tool name=\"HepMC\" version=\""${HEPMC2VER}"\">" >> ${xmlfile}
  tmppath=`find ${HEPMC2DIR} -type f -name libHepMC.so\*`
  tmpcnt=`echo ${tmppath} | grep -o "/" | grep -c "/"`
  tmppath=`echo ${tmppath} | cut -f 1-${tmpcnt} -d "/"`
  for LIB in `cd ${tmppath}; ls *.so | cut -f 1 -d "." | sed -e 's/lib//'; cd ${HDIR}`; do
    echo "    <lib name=\""${LIB}"\"/>" >> ${xmlfile}
  done
  echo "    <client>" >> ${xmlfile}
  echo "      <Environment name=\"HEPMC_BASE\" value=\""${HEPMC2DIR}"\"/>" >> ${xmlfile}
  echo "      <Environment name=\"LIBDIR\" default=\"\$HEPMC_BASE/lib\"/>" >> ${xmlfile}
  echo "      <Environment name=\"INCLUDE\" default=\"\$HEPMC_BASE/include\"/>" >> ${xmlfile}
  echo "    </client>" >> ${xmlfile}
  echo "    <runtime name=\"CMSSW_FWLITE_INCLUDE_PATH\" value=\"\$HEPMC_BASE/include\" type=\"path\"/>" >> ${xmlfile}
  echo "    <use name=\"CLHEP\"/>" >> ${xmlfile}
  echo "  </tool>" >> ${xmlfile}
  if [ ! "$PWD" = "${HDIR}" ]; then
    mv ${xmlfile} ${HDIR}/
  fi

  if [ ! "$CMSSW_BASE" = "" ]; then
    cd $CMSSW_BASE
    tmphmc=`scramv1 tool info hepmc | grep "HEPMC_BASE" | cut -f2 -d"="`
    tmpxml=`find $CMSSW_BASE/config -type f -name hepmc.xml -printf %h`
    echo " <I>"
    echo " <I> HEPMC version currently being used: "${tmphmc}
    echo " <I> ...defined in "${tmpxml}
    cd ${tmpxml}; tmpxml=$PWD; cd ${HDIR}
    echo " <I>"
    echo " <I> If you want to override this version with the freshly produced "${xmlfile}","
    echo " <I> ...please type the following commands:"
    echo " <I>"
    echo "       cd $CMSSW_BASE"
    echo "       scramv1 tool remove hepmc"
    echo "       cp ${HDIR}/${xmlfile} ${tmpxml}/"
    echo "       scramv1 setup hepmc"
    echo "       cd -"
    echo " <I>"
  fi

fi


echo " -> HEPMC2 installation directory is: "
echo "  "${HEPMC2IDIR}
