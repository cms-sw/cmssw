#!/bin/bash
#
#  file:        install_qd.sh
#  description: BASH script for the installation of the qd package,
#               can be used standalone or called from other scripts
#
#  author:      Markus Merschmeyer, RWTH Aachen University
#  date:        2013/05/23
#  version:     1.0
#

print_help() {
    echo "" && \
    echo "install_qd version 1.0" && echo && \
    echo "options: -v  version    define qd version ( "${QDVER}" )" && \
    echo "         -d  path       define qd installation directory" && \
    echo "                         -> ( "${IDIR}" )" && \
    echo "         -W  location   (web)location of qd tarball ( "${QDWEBLOCATION}" )" && \
    echo "         -S  filename   file name of qd tarball ( "${QDFILE}" )" && \
    echo "         -C  level      cleaning level of SHERPA installation ( "${LVLCLEAN}" )" && \
    echo "                         -> 0: nothing, 1: +objects (make clean)" && \
    echo "         -D             debug flag, compile with '-g' option ( "${FLGDEBUG}" )" && \
    echo "         -X             create XML file for tool override in CMSSW ( "${FLGXMLFL}" )" && \
    echo "         -Z             use multiple CPU cores if available ( "${FLGMCORE}" )" && \
    echo "         -K             keep qd source code tree after installation ( "${FGLKEEPT}" )" && \
    echo "         -h             display this help and exit" && echo
}


# save current path
HDIR=`pwd`


# dummy setup (if all options are missing)
IDIR="/tmp"                # installation directory
QDVER="2.3.13"             # qd version to be installed
QDWEBLOCATION=""           # (web)location of qd tarball
QDFILE=""                  # file name of qd tarball
LVLCLEAN=0                 # cleaning level (0-2)
FLGDEBUG="FALSE"           # debug flag for compilation
FLGXMLFL="FALSE"           # create XML tool definition file for SCRAM?
FGLKEEPT="FALSE"           # keep the source code tree?
FLGMCORE="FALSE"           # use multiple cores for compilation


# get & evaluate options
while getopts :v:d:W:S:C:DXZKh OPT
do
  case $OPT in
  v) QDVER=$OPTARG ;;
  d) IDIR=$OPTARG ;;
  W) QDWEBLOCATION=$OPTARG ;;
  S) QDFILE=$OPTARG ;;
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
      echo -n "install_qd: error: unrecognized option "
      if [ $OPTARG != "-" ]; then echo "'-$OPTARG'. try '-h'"
      else echo "'$1'. try '-h'"
      fi
      print_help && exit 1
    fi
#    shift 1
#    OPTIND=1
  esac
done


# set qd download location
if [ "$QDWEBLOCATION" = "" ]; then
  QDWEBLOCATION="http://crd.lbl.gov/~dhbailey/mpdist"
else
  if [ -e ${QDWEBLOCATION} ]; then   # is the location a local subdirectory?
    if [ -d ${QDWEBLOCATION} ]; then
      cd ${QDWEBLOCATION}; QDWEBLOCATION=`pwd`; cd ${HDIR}
    fi
  fi
fi
if [ "$QDFILE" = "" ]; then
  QDFILE="qd-${QDVER}.tar.gz"
fi


# make qd version a global variable
export QDVER=${QDVER}
# always use absolute path name...
cd ${IDIR}; IDIR=`pwd`

echo " qd installation: "
echo "  -> qd version: '"${QDVER}"'"
echo "  -> installation directory: '"${IDIR}"'"
echo "  -> qd location: '"${QDWEBLOCATION}"'"
echo "  -> qd file name: '"${QDFILE}"'"
echo "  -> cleaning level: '"${LVLCLEAN}"'"
echo "  -> debugging mode: '"${FLGDEBUG}"'"
echo "  -> CMSSW override: '"${FLGXMLFL}"'"
echo "  -> keep sources:   '"${FLGKEEPT}"'"
echo "  -> use multiple CPU cores: '"${FLGMCORE}"'"



# set path to local HEPMC2 installation
export QDDIR=${IDIR}"/qd-"${QDVER}
export QDIDIR=${IDIR}"/QD_"${QDVER}


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
COPTS="--enable-shared"
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


# download, extract compile/install qd
cd ${IDIR}
if [ ! -d ${QDIDIR} ]; then
  if [ `echo ${QDWEBLOCATION} | grep -c "http:"` -gt 0 ]; then
    echo " -> downloading qd "${QDVER}" from "${QDWEBLOCATION}/${QDFILE}
    wget ${QDWEBLOCATION}/${QDFILE}
  elif [ `echo ${QDWEBLOCATION} | grep -c "srm:"` -gt 0 ]; then
    echo " -> srm-copying qd "${QDVER}" from "${QDWEBLOCATION}/${QDFILE}
    srmcp ${QDWEBLOCATION}/${QDFILE} file:////${QDFILE}
  else
    echo " -> copying qd "${QDVER}" from "${QDWEBLOCATION}/${QDFILE}
    cp ${QDWEBLOCATION}/${QDFILE} ./
  fi
  tar -xzf ${QDFILE}
  if [ ! "$FLGKEEPT" = "TRUE" ]; then
    rm ${QDFILE}
  fi
  cd ${QDDIR}
#  if [ ! -e configure ]; then
#    ./bootstrap
#  fi

  echo " -> configuring qd with options "${COPTS} && \
  ./configure --prefix=${QDIDIR} ${momflag} ${lenflag} ${COPTS} && \
  echo " -> making qd with options "${POPTS} ${MOPTS} && \
  make ${POPTS} ${MOPTS} && \
  echo " -> installing qd with options "${MOPTS} && \
  make install ${MOPTS}
  if [ ${LVLCLEAN} -gt 0 ]; then 
    echo " -> cleaning up qd installation, level: "${LVLCLEAN}" ..."
    if [ ${LVLCLEAN} -ge 1 ]; then  # normal cleanup (objects)
      make clean
    fi
  fi
  cd ${HDIR}
  if [ "$FLGKEEPT" = "TRUE" ]; then
    echo "-> keeping source code..."
  else
    rm -rf ${QDDIR}
  fi
else
  echo " <W> path exists => using already installed qd"
fi
export QDDIR=${QDIDIR}
cd ${HDIR}


# create XML file fro SCRAM
if [ "${FLGXMLFL}" = "TRUE" ]; then
  xmlfile="qd_"${QDVER}".xml"
  echo " <I>"
  echo " <I> creating qd tool definition XML file"
  if [ -e ${xmlfile} ]; then rm ${xmlfile}; fi; touch ${xmlfile}
  echo "  <tool name=\"qd\" version=\""${QDVER}"\">" >> ${xmlfile}
  tmppath=`find ${QDDIR} -type f -name libqd.so\*`
#
	echo "QDDIR: "$QDDIR
	echo "TMPPATH: "$tmppath
#
  tmpcnt=`echo ${tmppath} | grep -o "/" | grep -c "/"`
  tmppath=`echo ${tmppath} | cut -f 1-${tmpcnt} -d "/"`
  for LIB in `cd ${tmppath}; ls *.so | cut -f 1 -d "." | sed -e 's/lib//'; cd ${HDIR}`; do
    echo "    <lib name=\""${LIB}"\"/>" >> ${xmlfile}
  done
  echo "    <client>" >> ${xmlfile}
  echo "      <Environment name=\"QD_BASE\" value=\""${QDDIR}"\"/>" >> ${xmlfile}
  echo "      <Environment name=\"LIBDIR\" default=\"\$QD_BASE/lib\"/>" >> ${xmlfile}
  echo "      <Environment name=\"INCLUDE\" default=\"\$QD_BASE/include\"/>" >> ${xmlfile}
  echo "    </client>" >> ${xmlfile}
#  echo "    <runtime name=\"CMSSW_FWLITE_INCLUDE_PATH\" value=\"\$QD_BASE/include\" type=\"path\"/>" >> ${xmlfile}
#  echo "    <use name=\"CLHEP\"/>" >> ${xmlfile}
  echo "  </tool>" >> ${xmlfile}
  if [ ! "$PWD" = "${HDIR}" ]; then
    mv ${xmlfile} ${HDIR}/
  fi

  if [ ! "$CMSSW_BASE" = "" ]; then
    cd $CMSSW_BASE
    tmpqd=`scramv1 tool info qd | grep "QD_BASE" | cut -f2 -d"="`
    tmpxml=`find $CMSSW_BASE/config -type f -name qd.xml -printf %h`
    echo " <I>"
    echo " <I> qd version currently being used: "${tmpqd}
    echo " <I> ...defined in "${tmpxml}
    cd ${tmpxml}; tmpxml=$PWD; cd ${HDIR}
    echo " <I>"
    echo " <I> If you want to override this version with the freshly produced "${xmlfile}","
    echo " <I> ...please type the following commands:"
    echo " <I>"
    echo "       cd $CMSSW_BASE"
    echo "       scramv1 tool remove qd"
    echo "       cp ${HDIR}/${xmlfile} ${tmpxml}/"
    echo "       scramv1 setup qd"
    echo "       cd -"
    echo " <I>"
  fi

fi


echo " -> qd installation directory is: "
echo "  "${QDIDIR}
