#!/bin/bash
#
#  file:        install_lhapdf.sh
#  description: BASH script for the installation of the LHAPDF package,
#               can be used standalone or called from other scripts
#
#  author:      Markus Merschmeyer, RWTH Aachen
#  date:        2008/03/25
#  version:     1.2
#

print_help() {
    echo "" && \
    echo "install_lhapdf version 1.2" && echo && \
    echo "options: -v  version    define LHAPDF version ( "${LHAPDFVER}" )" && \
    echo "         -d  path       define LHAPDF installation directory" && \
    echo "                         -> ( "${IDIR}" )" && \
    echo "         -f             require flags for 32-bit compilation ( "${FLAGS}" )" && \
    echo "         -n             use 'nopdf' version ( "${NOPDF}" )" && \
    echo "         -h             display this help and exit" && echo
}


# save current path
HDIR=`pwd`

# dummy setup (if all options are missing)
IDIR="/tmp"                # installation directory
LHAPDFVER="5.3.1"          # LHAPDF version to be installed
LHCFLAGS=" "               # compiler flags
LHMFLAGS=" "               # 'make' flags
FLAGS="FALSE"              # apply compiler/'make' flags
NOPDF="FALSE"              # install 'nopdf' version


# get & evaluate options
while getopts :v:d:fnh OPT
do
  case $OPT in
  v) LHAPDFVER=$OPTARG ;;
  d) IDIR=$OPTARG ;;
  f) FLAGS=TRUE ;;
  n) NOPDF=TRUE ;;
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


# make LHAPDF version a global variable
export LHAPDFVER=${LHAPDFVER}
# always use absolute path name...
cd ${IDIR}; IDIR=`pwd`


# set LHAPDF download location
LHAPDFWEBLOCATION="http://www.hepforge.org/archive/lhapdf"
if [ "$NOPDF" = "TRUE" ]; then
  LHAPDFFILE="lhapdf-"${LHAPDFVER}"-nopdf.tar.gz"
else
  LHAPDFFILE="lhapdf-"${LHAPDFVER}".tar.gz"
fi
export LHAPDFDIR=${IDIR}"/lhapdf-"${LHAPDFVER} # set path to local LHAPDF installation


# add compiler & linker flags
if [ "$FLAGS" = "TRUE" ]; then
    LHCFLAGS=${LHCFLAGS}" CFLAGS=-m32 FFLAGS=-m32 CXXFLAGS=-m32 LDFLAGS=-m32"
    LHMFLAGS=${LHMFLAGS}" CFLAGS=-m32 FFLAGS=-m32 CXXFLAGS=-m32 LDFLAGS=-m32"
fi


# download, extract compile/install LHAPDF
cd ${IDIR}
if [ ! -d ${LHAPDFDIR} ]; then
  echo " -> downloading LHAPDF "${LHAPDFVER}" from "${LHAPDFWEBLOCATION}/${LHAPDFFILE}
  wget ${LHAPDFWEBLOCATION}/${LHAPDFFILE}
  tar -xzf ${LHAPDFFILE}
  rm ${LHAPDFFILE}
  cd ${LHAPDFDIR}
  echo " -> configuring LHAPDF with flags: "${LHCFLAGS}
  ./configure  --prefix=${LHAPDFDIR} ${LHCFLAGS}
  echo " -> installing LHAPDF with flags: "${LHMFLAGS}
  make ${LHMFLAGS}
  make install ${LHMFLAGS}
else
  echo " <W> path exists => using already installed LHAPDF"
fi
cd ${HDIR}


echo " -> LHAPDF installation directory is: "
echo "  "${LHAPDFDIR}


