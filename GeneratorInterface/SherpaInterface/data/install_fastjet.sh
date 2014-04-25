#!/bin/bash
#
#  file:        install_fastjet.sh
#  description: BASH script for the installation of the fastjet package,
#               can be used standalone or called from other scripts
#
#  author:      Markus Merschmeyer, RWTH Aachen University
#  date:        2013/07/29
#  version:     1.0
#

print_help() {
    echo "" && \
    echo "install_fastjet version 1.0" && echo && \
    echo "options: -v  version    define fastjet version ( "${FJVER}" )" && \
    echo "         -d  path       define fastjet installation directory" && \
    echo "                         -> ( "${IDIR}" )" && \
    echo "         -W  location   (web)location of fastjet tarball ( "${FJWEBLOCATION}" )" && \
    echo "         -S  filename   file name of fastjet tarball ( "${FJFILE}" )" && \
    echo "         -C  level      cleaning level of SHERPA installation ( "${LVLCLEAN}" )" && \
    echo "                         -> 0: nothing, 1: +objects (make clean)" && \
    echo "         -D             debug flag, compile with '-g' option ( "${FLGDEBUG}" )" && \
    echo "         -X             create XML file for tool override in CMSSW ( "${FLGXMLFL}" )" && \
    echo "         -Z             use multiple CPU cores if available ( "${FLGMCORE}" )" && \
    echo "         -K             keep fastjet source code tree after installation ( "${FGLKEEPT}" )" && \
    echo "         -h             display this help and exit" && echo
}


# save current path
HDIR=`pwd`


# dummy setup (if all options are missing)
IDIR="/tmp"                # installation directory
FJVER="2.4.5"              # version to be installed
FJWEBLOCATION=""           # (web)location of fastjet tarball
FJFILE=""                  # file name of fastjet tarball
LVLCLEAN=0                 # cleaning level (0-2)
FLGDEBUG="FALSE"           # debug flag for compilation
FLGXMLFL="FALSE"           # create XML tool definition file for SCRAM?
FGLKEEPT="FALSE"           # keep the source code tree?
FLGMCORE="FALSE"           # use multiple cores for compilation


# get & evaluate options
while getopts :v:d:W:S:C:DXZKh OPT
do
  case $OPT in
  v) FJVER=$OPTARG ;;
  d) IDIR=$OPTARG ;;
  W) FJWEBLOCATION=$OPTARG ;;
  S) FJFILE=$OPTARG ;;
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
      echo -n "install_fastjet: error: unrecognized option "
      if [ $OPTARG != "-" ]; then echo "'-$OPTARG'. try '-h'"
      else echo "'$1'. try '-h'"
      fi
      print_help && exit 1
    fi
#    shift 1
#    OPTIND=1
  esac
done


# set fastjet download location
if [ "$FJWEBLOCATION" = "" ]; then
  FJWEBLOCATION="http://fastjet.fr/repo"
else
  if [ -e ${FJWEBLOCATION} ]; then   # is the location a local subdirectory?
    if [ -d ${FJWEBLOCATION} ]; then
      cd ${FJWEBLOCATION}; FJWEBLOCATION=`pwd`; cd ${HDIR}
    fi
  fi
fi
if [ "$FJFILE" = "" ]; then
  FJFILE="fastjet-${FJVER}.tar.gz"
fi


# make fastjet version a global variable
export FJVER=${FJVER}
# always use absolute path name...
cd ${IDIR}; IDIR=`pwd`

echo " fastjet installation: "
echo "  -> fastjet version: '"${FJVER}"'"
echo "  -> installation directory: '"${IDIR}"'"
echo "  -> fastjet location: '"${FJWEBLOCATION}"'"
echo "  -> fastjet file name: '"${FJFILE}"'"
echo "  -> cleaning level: '"${LVLCLEAN}"'"
echo "  -> debugging mode: '"${FLGDEBUG}"'"
echo "  -> CMSSW override: '"${FLGXMLFL}"'"
echo "  -> keep sources:   '"${FLGKEEPT}"'"
echo "  -> use multiple CPU cores: '"${FLGMCORE}"'"



# set path to local HEPMC2 installation
export FJDIR=${IDIR}"/fastjet-"${FJVER}
export FJIDIR=${IDIR}"/FJ_"${FJVER}


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


# download, extract compile/install fastjet
cd ${IDIR}
if [ ! -d ${FJIDIR} ]; then
  if [ `echo ${FJWEBLOCATION} | grep -c "http:"` -gt 0 ]; then
    echo " -> downloading fastjet "${FJVER}" from "${FJWEBLOCATION}/${FJFILE}
    wget ${FJWEBLOCATION}/${FJFILE}
  elif [ `echo ${FJWEBLOCATION} | grep -c "srm:"` -gt 0 ]; then
    echo " -> srm-copying fastjet "${FJVER}" from "${FJWEBLOCATION}/${FJFILE}
    srmcp ${FJWEBLOCATION}/${FJFILE} file:////${FJFILE}
  else
    echo " -> copying fastjet "${FJVER}" from "${FJWEBLOCATION}/${FJFILE}
    cp ${FJWEBLOCATION}/${FJFILE} ./
  fi
  tar -xzf ${FJFILE}
  if [ ! "$FLGKEEPT" = "TRUE" ]; then
    rm ${FJFILE}
  fi
  cd ${FJDIR}

  echo " -> configuring fastjet with options "${COPTS} && \
  ./configure --prefix=${FJIDIR} ${momflag} ${lenflag} ${COPTS} && \
  echo " -> making fastjet with options "${POPTS} ${MOPTS} && \
  make ${POPTS} ${MOPTS} && \
  echo " -> installing fastjet with options "${MOPTS} && \
  make install ${MOPTS}
  if [ ${LVLCLEAN} -gt 0 ]; then 
    echo " -> cleaning up fastjet installation, level: "${LVLCLEAN}" ..."
    if [ ${LVLCLEAN} -ge 1 ]; then  # normal cleanup (objects)
      make clean
    fi
  fi
  cd ${HDIR}
  if [ "$FLGKEEPT" = "TRUE" ]; then
    echo "-> keeping source code..."
  else
    rm -rf ${FJDIR}
  fi
else
  echo " <W> path exists => using already installed fastjet"
fi
export FJDIR=${FJIDIR}
cd ${HDIR}


# create XML file fro SCRAM
if [ "${FLGXMLFL}" = "TRUE" ]; then
  xmlfile="fastjet_"${FJVER}".xml"
  echo " <I>"
  echo " <I> creating fastjet tool definition XML file"
  if [ -e ${xmlfile} ]; then rm ${xmlfile}; fi; touch ${xmlfile}
  echo "  <tool name=\"fastjet\" version=\""${FJVER}"\">" >> ${xmlfile}
  tmppath=`find ${FJDIR} -type f -name libfastjet.so\*`
#
	echo "FJDIR: "$FJDIR
	echo "TMPPATH: "$tmppath
#
  tmpcnt=`echo ${tmppath} | grep -o "/" | grep -c "/"`
  tmppath=`echo ${tmppath} | cut -f 1-${tmpcnt} -d "/"`
  for LIB in `cd ${tmppath}; ls *.so | cut -f 1 -d "." | sed -e 's/lib//'; cd ${HDIR}`; do
    echo "    <lib name=\""${LIB}"\"/>" >> ${xmlfile}
  done
  echo "    <client>" >> ${xmlfile}
  echo "      <Environment name=\"FJ_BASE\" value=\""${FJDIR}"\"/>" >> ${xmlfile}
  echo "      <Environment name=\"LIBDIR\" default=\"\$FJ_BASE/lib\"/>" >> ${xmlfile}
  echo "      <Environment name=\"INCLUDE\" default=\"\$FJ_BASE/include\"/>" >> ${xmlfile}
  echo "    </client>" >> ${xmlfile}
  echo "  </tool>" >> ${xmlfile}
  if [ ! "$PWD" = "${HDIR}" ]; then
    mv ${xmlfile} ${HDIR}/
  fi

  if [ ! "$CMSSW_BASE" = "" ]; then
    cd $CMSSW_BASE
    tmpom=`scramv1 tool info fastjet | grep "FJ_BASE" | cut -f2 -d"="`
    tmpxml=`find $CMSSW_BASE/config -type f -name qd.xml -printf %h`
    echo " <I>"
    echo " <I> fastjet version currently being used: "${tmpom}
    echo " <I> ...defined in "${tmpxml}
    cd ${tmpxml}; tmpxml=$PWD; cd ${HDIR}
    echo " <I>"
    echo " <I> If you want to override this version with the freshly produced "${xmlfile}","
    echo " <I> ...please type the following commands:"
    echo " <I>"
    echo "       cd $CMSSW_BASE"
    echo "       scramv1 tool remove fastjet"
    echo "       cp ${HDIR}/${xmlfile} ${tmpxml}/"
    echo "       scramv1 setup fastjet"
    echo "       cd -"
    echo " <I>"
  fi

fi


echo " -> fastjet installation directory is: " &&\
echo "  "${FJIDIR} && \
echo "" &&\
echo "   + please make sure to add (prepend)" &&\
echo "       "${FJIDIR}"/lib" &&\
echo "      to LD_LIBRARY_PATH environment variable" &&\
echo "" &&\
echo ""


## update LD_LIBRARY_PATH
