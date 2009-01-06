#!/bin/bash
#
#  file:        install_lhapdf.sh
#  description: BASH script for the installation of the LHAPDF package,
#               can be used standalone or called from other scripts
#
#  author:      Markus Merschmeyer, RWTH Aachen
#  date:        2009/01/05
#  version:     2.3
#

print_help() {
    echo "" && \
    echo "install_lhapdf version 2.3" && echo && \
    echo "options: -v  version    define LHAPDF version ( "${LHAPDFVER}" )" && \
    echo "         -d  path       define LHAPDF installation directory" && \
    echo "                         -> ( "${IDIR}" )" && \
    echo "         -f             require flags for 32-bit compilation ( "${FLAGS}" )" && \
    echo "         -n             use 'nopdf' version ( "${NOPDF}" )" && \
    echo "         -l             user '--enable-low-memory' option ( "${LOWMEM}")" && \
    echo "         -W  location   (web)location of LHAPDF tarball ( "${LHAPDFWEBLOCATION}" )" && \
    echo "         -S  filename   file name of LHAPDF tarball ( "${LHAPDFFILE}" )" && \
    echo "         -C  level      cleaning level of SHERPA installation ("${LVLCLEAN}" )" && \
    echo "                         -> 0: nothing, 1: +objects (make clean)" && \
    echo "         -D             debug flag, compile with '-g' option ("${FLGDEBUG}" )" && \
    echo "         -X             create XML file for tool override in CMSSW" && \
    echo "         -h             display this help and exit" && echo
}


# save current path
HDIR=`pwd`


# dummy setup (if all options are missing)
IDIR="/tmp"                # installation directory
LHAPDFVER="5.6.0"          # LHAPDF version to be installed
FLAGS="FALSE"              # apply compiler/'make' flags
NOPDF="FALSE"              # install 'nopdf' version
LOWMEM="FALSE"             # use 'low-memory' option
LHAPDFWEBLOCATION=""       # (web)location of LHAPDF tarball
LHAPDFFILE=""              # file name of LHAPDF tarball
LVLCLEAN=0                 # cleaning level (0-2)
FLGDEBUG="FALSE"           # debug flag for compilation
FLGXMLFL="FALSE"           # create XML tool definition file for SCRAM?


# get & evaluate options
while getopts :v:d:W:S:C:fnlDXh OPT
do
  case $OPT in
  v) LHAPDFVER=$OPTARG ;;
  d) IDIR=$OPTARG ;;
  f) FLAGS=TRUE ;;
  n) NOPDF=TRUE ;;
  l) LOWMEM=TRUE ;;
  W) LHAPDFWEBLOCATION=$OPTARG ;;
  S) LHAPDFFILE=$OPTARG ;;
  C) LVLCLEAN=$OPTARG ;;
  D) FLGDEBUG=TRUE ;;
  X) FLGXMLFL=TRUE ;;
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
echo "  -> CMSSW override: '"${FLGXMLFL}"'"


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


# add compiler & linker flags
COPTS=""
MOPTS=""
if [ "$FLAGS" = "TRUE" ]; then
    COPTS=${COPTS}" CFLAGS=-m32 FFLAGS=-m32 CXXFLAGS=-m32 LDFLAGS=-m32"
    MOPTS=${MOPTS}" CFLAGS=-m32 FFLAGS=-m32 CXXFLAGS=-m32 LDFLAGS=-m32"
fi

# download, extract compile/install LHAPDF
cd ${IDIR}
#if [ ! -d ${LHAPDFDIR} ]; then
if [ ! -d ${LHAPDFIDIR} ]; then
  if [ "${LOWMEM}" = "TRUE" ]; then
    COPTS=${COPTS}" --enable-low-memory"
  fi
  echo " -> downloading LHAPDF "${LHAPDFVER}" from "${LHAPDFWEBLOCATION}/${LHAPDFFILE}
  wget ${LHAPDFWEBLOCATION}/${LHAPDFFILE}
  tar -xzf ${LHAPDFFILE}
  rm ${LHAPDFFILE}
  cd ${LHAPDFDIR}
  echo " -> configuring LHAPDF with options: "${COPTS}
  ./configure --prefix=${LHAPDFIDIR} ${COPTS}
  echo " -> making LHAPDF with options "${MOPTS}
  make ${MOPTS}
  echo " -> installing LHAPDF with options "${MOPTS}
  make install ${MOPTS}
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
export LHAPDFDIR=${LHAPDFIDIR}
cd ${HDIR}


# create XML file fro SCRAM
if [ "${FLGXMLFL}" = "TRUE" ]; then
  xmlfile=lhapdf.xml
  echo " <I> creating LHAPDF tool definition XML file"
  if [ -e ${xmlfile} ]; then rm ${xmlfile}; fi; touch ${xmlfile}
  echo "  <tool name=\"lhapdf\" version=\""${LHAPDFVER}"\">" >> ${xmlfile}
  tmppath=`find ${LHAPDFDIR} -type f -name libLHAPDF.so\*`
  tmpcnt=`echo ${tmppath} | grep -o "/" | grep -c "/"`
  tmppath=`echo ${tmppath} | cut -f 0-${tmpcnt} -d "/"`
  for LIB in `cd ${tmppath}; ls *.so | cut -f 1 -d "." | sed -e 's/lib//'; cd ${HDIR}`; do
    echo "    <lib name=\""${LIB}"\"/>" >> ${xmlfile}
  done
  echo "    <client>" >> ${xmlfile}
  echo "      <Environment name=\"LHAPDF_BASE\" value=\""${LHAPDFDIR}"\"/>" >> ${xmlfile}
  echo "      <Environment name=\"LIBDIR\" default=\"\$LHAPDF_BASE/lib\"/>" >> ${xmlfile}
  echo "      <Environment name=\"LHAPATH\" default=\"\$LHAPDF_BASE/share/lhapdf/PDFsets\"/>" >> ${xmlfile}
  echo "    </client>" >> ${xmlfile}
  echo "    <runtime name=\"LHAPATH\" value=\"\$LHAPDF_BASE/share/lhapdf/PDFsets\" type=\"path\"/>" >> ${xmlfile}
  echo "    <use name=\"f77compiler\"/>" >> ${xmlfile}
  echo "  </tool>" >> ${xmlfile}
  mv ${xmlfile} ${HDIR}/
fi


echo " -> LHAPDF installation directory is: "
echo "  "${LHAPDFIDIR}
