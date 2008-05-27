#!/bin/bash
#
#  file:        install_hepmc2.sh
#  description: BASH script for the installation of the HepMC2 package,
#               can be used standalone or called from other scripts
#
#  author:      Markus Merschmeyer, RWTH Aachen
#  date:        2008/03/25
#  version:     1.2
#

print_help() {
    echo "" && \
    echo "install_hepmc2 version 1.2" && echo && \
    echo "options: -v  version    define HepMC2 version ( "${HEPMC2VER}" )" && \
    echo "         -d  path       define HepMC2 installation directory" && \
    echo "                         -> ( "${IDIR}" )" && \
    echo "         -f             require flags for 32-bit compilation ( "${FLAGS}" )" && \
    echo "         -h             display this help and exit" && echo
}


# save current path
HDIR=`pwd`

# dummy setup (if all options are missing)
IDIR="/tmp"                # installation directory
HEPMC2VER="2.01.10"        # HepMC2 version  to be installed
HMCFLAGS=" "               # compiler flags
HMMFLAGS=" "               # 'make' flags
FLAGS="FALSE"              # apply compiler/'make' flags


# get & evaluate options
while getopts :v:d:fh OPT
do
  case $OPT in
  v) HEPMC2VER=$OPTARG ;;
  d) IDIR=$OPTARG ;;
  f) FLAGS=TRUE ;;
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


# make HEPMC2 version a global variable
export HEPMC2VER=${HEPMC2VER}
# always use absolute path name...
cd ${IDIR}; IDIR=`pwd`


# set HEPMC2 download location
HEPMC2WEBLOCATION="http://lcgapp.cern.ch/project/simu/HepMC/download"
HEPMC2FILE="HepMC-"${HEPMC2VER}".tar.gz"
export HEPMC2DIR=${IDIR}"/HepMC-"${HEPMC2VER} # set path to local HEPMC2 installation


# add compiler & linker flags
if [ "$FLAGS" = "TRUE" ]; then
    HMCFLAGS=${HMCFLAGS}" CFLAGS=-m32 FFLAGS=-m32 CXXFLAGS=-m32 LDFLAGS=-m32"
    HMMFLAGS=${HMMFLAGS}" CFLAGS=-m32 FFLAGS=-m32 CXXFLAGS=-m32 LDFLAGS=-m32"
fi


# download, extract compile/install HEPMC2
cd ${IDIR}
if [ ! -d ${HEPMC2DIR} ]; then
  echo " -> downloading HepMC2 "${HEPMC2VER}" from "${HEPMC2WEBLOCATION}/${HEPMC2FILE}
  wget ${HEPMC2WEBLOCATION}/${HEPMC2FILE}
  tar -xzf ${HEPMC2FILE}
  rm ${HEPMC2FILE}
  cd ${HEPMC2DIR}
  echo " -> configuring HepMC2 with flags: "${HMCFLAGS}
  ./configure  --prefix=${HEPMC2DIR} ${HMCFLAGS}
  echo " -> installing HepMC2 with flags: "${HMMFLAGS}
  make install ${HMMFLAGS}
else
  echo " <W> path exists => using already installed HepMC2"
fi
cd ${HDIR}


echo " -> HEPMC2 installation directory is: "
echo "  "${HEPMC2DIR}


