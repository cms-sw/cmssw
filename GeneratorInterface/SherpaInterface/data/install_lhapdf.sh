#!/bin/bash
#
#  file:        install_lhapdf.sh
#  description: BASH script for the installation of the LHAPDF package,
#               can be used standalone or called from other scripts
#
#  author:      Markus Merschmeyer, RWTH Aachen
#  date:        2008/09/19
#  version:     2.1
#

print_help() {
    echo "" && \
    echo "install_lhapdf version 2.1" && echo && \
    echo "options: -v  version    define LHAPDF version ( "${LHAPDFVER}" )" && \
    echo "         -d  path       define LHAPDF installation directory" && \
    echo "                         -> ( "${IDIR}" )" && \
    echo "         -f             require flags for 32-bit compilation ( "${FLAGS}" )" && \
    echo "         -n             use 'nopdf' version ( "${NOPDF}" )" && \
    echo "         -W  location   (web)location of LHAPDF tarball ( "${LHAPDFWEBLOCATION}" )" && \
    echo "         -S  filename   file name of LHAPDF tarball ( "${LHAPDFFILE}" )" && \
    echo "         -C  level      cleaning level of SHERPA installation ("${LVLCLEAN}" )" && \
    echo "                         -> 0: nothing, 1: +objects (make clean)" && \
    echo "         -D             debug flag, compile with '-g' option ("${FLGDEBUG}" )" && \
    echo "         -h             display this help and exit" && echo
}


# save current path
HDIR=`pwd`


# dummy setup (if all options are missing)
IDIR="/tmp"                # installation directory
LHAPDFVER="5.3.1"          # LHAPDF version to be installed
FLAGS="FALSE"              # apply compiler/'make' flags
NOPDF="FALSE"              # install 'nopdf' version
LHAPDFWEBLOCATION=""       # (web)location of LHAPDF tarball
LHAPDFFILE=""              # file name of LHAPDF tarball
LVLCLEAN=0                 # cleaning level (0-2)
FLGDEBUG="FALSE"           # debug flag for compilation


# get & evaluate options
while getopts :v:d:W:S:C:fnDh OPT
do
  case $OPT in
  v) LHAPDFVER=$OPTARG ;;
  d) IDIR=$OPTARG ;;
  f) FLAGS=TRUE ;;
  n) NOPDF=TRUE ;;
  W) LHAPDFWEBLOCATION=$OPTARG ;;
  S) LHAPDFFILE=$OPTARG ;;
  C) LVLCLEAN=$OPTARG ;;
  D) FLGDEBUG=TRUE ;;
  h) print_help && exit 0 ;;
  \?)
    shift `expr $OPTIND - 1`
    if [ "$1" = "--help" ]; then print_help && exit 0;
    else 
      echo -n "install_lhapdf: error: unrecognized option "
      if [ $OPTARG != "-" ]; then echo "'-$OPTARG'. try '-h'"
      else echo "'$1'. try '-h'"
      fi
      print_help && exit 1
    fi
#    shift 1
#    OPTIND=1
  esac
done


# set LHAPDF download location
if [ "$LHAPDFWEBLOCATION" = "" ]; then
  LHAPDFWEBLOCATION="http://www.hepforge.org/archive/lhapdf"
fi
if [ "$LHAPDFFILE" = "" ]; then
  if [ "$NOPDF" = "TRUE" ]; then
    LHAPDFFILE="lhapdf-"${LHAPDFVER}"-nopdf.tar.gz"
  else
    LHAPDFFILE="lhapdf-"${LHAPDFVER}".tar.gz"
  fi
fi


# make LHAPDF version a global variable
export LHAPDFVER=${LHAPDFVER}
# always use absolute path name...
cd ${IDIR}; IDIR=`pwd`

echo " LHAPDF installation: "
echo "  -> LHAPDF version: '"${LHAPDFVER}"'"
echo "  -> installation directory: '"${IDIR}"'"
echo "  -> flags: '"${FLAGS}"'"
echo "  -> no PDF version: '"${NOPDF}"'"
echo "  -> LHAPDF location: '"${LHAPDFWEBLOCATION}"'"
echo "  -> LHAPDF file name: '"${LHAPDFFILE}"'"
echo "  -> cleaning level: '"${LVLCLEAN}"'"
echo "  -> debugging mode: '"${FLGDEBUG}"'"


# set path to local LHAPDF installation
export LHAPDFDIR=${IDIR}"/lhapdf-"${LHAPDFVER}
export LHAPDFIDIR=${IDIR}"/LHAPDF_"${LHAPDFVER}


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


# download, extract compile/install LHAPDF
cd ${IDIR}
#if [ ! -d ${LHAPDFDIR} ]; then
if [ ! -d ${LHAPDFIDIR} ]; then
  echo " -> downloading LHAPDF "${LHAPDFVER}" from "${LHAPDFWEBLOCATION}/${LHAPDFFILE}
  wget ${LHAPDFWEBLOCATION}/${LHAPDFFILE}
  tar -xzf ${LHAPDFFILE}
  rm ${LHAPDFFILE}
  cd ${LHAPDFDIR}
  echo " -> configuring LHAPDF"
  ./configure --prefix=${LHAPDFIDIR}
  echo " -> making LHAPDF"
  make
  echo " -> installing LHAPDF"
  make install
  if [ ${LVLCLEAN} -gt 0 ]; then 
    echo " -> cleaning up LHAPDF installation, level: "${LVLCLEAN}" ..."
    if [ ${LVLCLEAN} -ge 1 ]; then  # normal cleanup (objects)
      make clean
    fi
  fi
else
  echo " <W> path exists => using already installed LHAPDF"
fi
rm -rf ${LHAPDFDIR}
cd ${HDIR}


echo " -> LHAPDF installation directory is: "
echo "  "${LHAPDFIDIR}
