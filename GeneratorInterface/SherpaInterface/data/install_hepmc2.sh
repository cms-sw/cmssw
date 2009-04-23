#!/bin/bash
#
#  file:        install_hepmc2.sh
#  description: BASH script for the installation of the HepMC2 package,
#               can be used standalone or called from other scripts
#
#  author:      Markus Merschmeyer, RWTH Aachen
#  date:        2008/09/19
#  version:     2.1
#

print_help() {
    echo "" && \
    echo "install_hepmc2 version 2.1" && echo && \
    echo "options: -v  version    define HepMC2 version ( "${HEPMC2VER}" )" && \
    echo "         -d  path       define HepMC2 installation directory" && \
    echo "                         -> ( "${IDIR}" )" && \
    echo "         -f             require flags for 32-bit compilation ( "${FLAGS}" )" && \
    echo "         -W  location   (web)location of HepMC2 tarball ( "${HEPMC2WEBLOCATION}" )" && \
    echo "         -S  filename   file name of HepMC2 tarball ( "${HEPMC2FILE}" )" && \
    echo "         -C  level      cleaning level of SHERPA installation ("${LVLCLEAN}" )" && \
    echo "                         -> 0: nothing, 1: +objects (make clean)" && \
    echo "         -D             debug flag, compile with '-g' option ("${FLGDEBUG}" )" && \
    echo "         -h             display this help and exit" && echo
}


# save current path
HDIR=`pwd`


# dummy setup (if all options are missing)
IDIR="/tmp"                # installation directory
HEPMC2VER="2.01.10"        # HepMC2 version  to be installed
FLAGS="FALSE"              # apply compiler/'make' flags
HEPMC2WEBLOCATION=""       # (web)location of HEPMC2 tarball
HEPMC2FILE=""              # file name of HEPMC2 tarball
LVLCLEAN=0                 # cleaning level (0-2)
FLGDEBUG="FALSE"           # debug flag for compilation


# get & evaluate options
while getopts :v:d:W:S:C:fDh OPT
do
  case $OPT in
  v) HEPMC2VER=$OPTARG ;;
  d) IDIR=$OPTARG ;;
  f) FLAGS=TRUE ;;
  W) HEPMC2WEBLOCATION=$OPTARG ;;
  S) HEPMC2FILE=$OPTARG ;;
  C) LVLCLEAN=$OPTARG ;;
  D) FLGDEBUG=TRUE ;;
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
echo "  -> flags: '"${FLAGS}"'"
echo "  -> HepMC2 location: '"${HEPMC2WEBLOCATION}"'"
echo "  -> HepMC2 file name: '"${HEPMC2FILE}"'"
echo "  -> cleaning level: '"${LVLCLEAN}"'"
echo "  -> debugging mode: '"${FLGDEBUG}"'"

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
echo "CFLAGS   (old):  "$CFLAGS
echo "FFLAGS   (old):  "$FFLAGS
echo "CXXFLAGS (old):  "$CXXFLAGS
echo "LDFLAGS  (old):  "$LDFLAGS
CF32BIT=""
if [ "$FLAGS" = "TRUE" ]; then
  CF32BIT="-m32"
  export CFLAGS=${CFLAGS}" "${CF32BIT}
  export FFLAGS=${FFLAGS}" "${CF32BIT}
  export CXXFLAGS=${CXXFLAGS}" "${CF32BIT}
  export LDFLAGS=${LDFLAGS}" "${CF32BIT}
fi
CFDEBUG=""
if [ "$FLGDEBUG" = "TRUE" ]; then
  CFDEBUG="-g"
  export CFLAGS=${CFLAGS}" "${CFDEBUG}
  export FFLAGS=${FFLAGS}" "${CFDEBUG}
  export CXXFLAGS=${CXXFLAGS}" "${CFDEBUG}
fi
echo "CFLAGS   (new):  "$CFLAGS
echo "FFLAGS   (new):  "$FFLAGS
echo "CXXFLAGS (new):  "$CXXFLAGS
echo "LDFLAGS  (new):  "$LDFLAGS


# download, extract compile/install HEPMC2
cd ${IDIR}
#if [ ! -d ${HEPMC2DIR} ]; then
if [ ! -d ${HEPMC2IDIR} ]; then
  echo " -> downloading HepMC2 "${HEPMC2VER}" from "${HEPMC2WEBLOCATION}/${HEPMC2FILE}
  wget ${HEPMC2WEBLOCATION}/${HEPMC2FILE}
  tar -xzf ${HEPMC2FILE}
  rm ${HEPMC2FILE}
  cd ${HEPMC2DIR}
  echo " -> configuring HepMC2"
  ./configure --prefix=${HEPMC2IDIR} ${momflag} ${lenflag}
  echo " -> making HepMC2"
  make
  echo " -> installing HepMC2"
  make install
  if [ ${LVLCLEAN} -gt 0 ]; then 
    echo " -> cleaning up HEPMC2 installation, level: "${LVLCLEAN}" ..."
    if [ ${LVLCLEAN} -ge 1 ]; then  # normal cleanup (objects)
      make clean
    fi
  fi
else
  echo " <W> path exists => using already installed HepMC2"
fi
rm -rf ${HEPMC2DIR}
cd ${HDIR}


echo " -> HEPMC2 installation directory is: "
echo "  "${HEPMC2IDIR}
