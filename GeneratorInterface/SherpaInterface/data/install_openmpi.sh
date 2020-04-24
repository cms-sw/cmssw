#!/bin/bash
#
#  file:        install_openmpi.sh
#  description: BASH script for the installation of the openmpi package,
#               can be used standalone or called from other scripts
#
#  author:      Markus Merschmeyer, RWTH Aachen University
#  date:        2013/07/22
#  version:     1.0
#

print_help() {
    echo "" && \
    echo "install_openmpi version 1.0" && echo && \
    echo "options: -v  version    define openmpi version ( "${OMVER}" )" && \
    echo "         -d  path       define openmpi installation directory" && \
    echo "                         -> ( "${IDIR}" )" && \
    echo "         -W  location   (web)location of openmpi tarball ( "${OMWEBLOCATION}" )" && \
    echo "         -S  filename   file name of openmpi tarball ( "${OMFILE}" )" && \
    echo "         -C  level      cleaning level of SHERPA installation ( "${LVLCLEAN}" )" && \
    echo "                         -> 0: nothing, 1: +objects (make clean)" && \
    echo "         -D             debug flag, compile with '-g' option ( "${FLGDEBUG}" )" && \
    echo "         -X             create XML file for tool override in CMSSW ( "${FLGXMLFL}" )" && \
    echo "         -Z             use multiple CPU cores if available ( "${FLGMCORE}" )" && \
    echo "         -K             keep openmpi source code tree after installation ( "${FGLKEEPT}" )" && \
    echo "         -h             display this help and exit" && echo
}


# save current path
HDIR=`pwd`


# dummy setup (if all options are missing)
IDIR="/tmp"                # installation directory
OMVER="1.6.5"              # version to be installed
OMWEBLOCATION=""           # (web)location of openmpi tarball
OMFILE=""                  # file name of openmpi tarball
LVLCLEAN=0                 # cleaning level (0-2)
FLGDEBUG="FALSE"           # debug flag for compilation
FLGXMLFL="FALSE"           # create XML tool definition file for SCRAM?
FGLKEEPT="FALSE"           # keep the source code tree?
FLGMCORE="FALSE"           # use multiple cores for compilation


# get & evaluate options
while getopts :v:d:W:S:C:DXZKh OPT
do
  case $OPT in
  v) OMVER=$OPTARG ;;
  d) IDIR=$OPTARG ;;
  W) OMWEBLOCATION=$OPTARG ;;
  S) OMFILE=$OPTARG ;;
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
      echo -n "install_openmpi: error: unrecognized option "
      if [ $OPTARG != "-" ]; then echo "'-$OPTARG'. try '-h'"
      else echo "'$1'. try '-h'"
      fi
      print_help && exit 1
    fi
#    shift 1
#    OPTIND=1
  esac
done


# set openmpi download location
if [ "$OMWEBLOCATION" = "" ]; then
  OMWEBLOCATION="http://www.open-mpi.org/software/ompi/v1.6/downloads"
else
  if [ -e ${OMWEBLOCATION} ]; then   # is the location a local subdirectory?
    if [ -d ${OMWEBLOCATION} ]; then
      cd ${OMWEBLOCATION}; OMWEBLOCATION=`pwd`; cd ${HDIR}
    fi
  fi
fi
if [ "$OMFILE" = "" ]; then
  OMFILE="openmpi-${OMVER}.tar.gz"
fi


# make openmpi version a global variable
export OMVER=${OMVER}
# always use absolute path name...
cd ${IDIR}; IDIR=`pwd`

echo " openmpi installation: "
echo "  -> openmpi version: '"${OMVER}"'"
echo "  -> installation directory: '"${IDIR}"'"
echo "  -> openmpi location: '"${OMWEBLOCATION}"'"
echo "  -> openmpi file name: '"${OMFILE}"'"
echo "  -> cleaning level: '"${LVLCLEAN}"'"
echo "  -> debugging mode: '"${FLGDEBUG}"'"
echo "  -> CMSSW override: '"${FLGXMLFL}"'"
echo "  -> keep sources:   '"${FLGKEEPT}"'"
echo "  -> use multiple CPU cores: '"${FLGMCORE}"'"



# set path to local HEPMC2 installation
export OMDIR=${IDIR}"/openmpi-"${OMVER}
export OMIDIR=${IDIR}"/OM_"${OMVER}


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


# download, extract compile/install openmpi
cd ${IDIR}
if [ ! -d ${OMIDIR} ]; then
  if [ `echo ${OMWEBLOCATION} | grep -c "http:"` -gt 0 ]; then
    echo " -> downloading openmpi "${OMVER}" from "${OMWEBLOCATION}/${OMFILE}
    wget ${OMWEBLOCATION}/${OMFILE}
  elif [ `echo ${OMWEBLOCATION} | grep -c "srm:"` -gt 0 ]; then
    echo " -> srm-copying openmpi "${OMVER}" from "${OMWEBLOCATION}/${OMFILE}
    srmcp ${OMWEBLOCATION}/${OMFILE} file:////${OMFILE}
  else
    echo " -> copying openmpi "${OMVER}" from "${OMWEBLOCATION}/${OMFILE}
    cp ${OMWEBLOCATION}/${OMFILE} ./
  fi
  tar -xzf ${OMFILE}
  if [ ! "$FLGKEEPT" = "TRUE" ]; then
    rm ${OMFILE}
  fi
  cd ${OMDIR}

  echo " -> configuring openmpi with options "${COPTS} && \
  ./configure --prefix=${OMIDIR} ${momflag} ${lenflag} ${COPTS} && \
  echo " -> making openmpi with options "${POPTS} ${MOPTS} && \
  make ${POPTS} ${MOPTS} && \
  echo " -> installing openmpi with options "${MOPTS} && \
  make install ${MOPTS}
  if [ ${LVLCLEAN} -gt 0 ]; then 
    echo " -> cleaning up openmpi installation, level: "${LVLCLEAN}" ..."
    if [ ${LVLCLEAN} -ge 1 ]; then  # normal cleanup (objects)
      make clean
    fi
  fi
  cd ${HDIR}
  if [ "$FLGKEEPT" = "TRUE" ]; then
    echo "-> keeping source code..."
  else
    rm -rf ${OMDIR}
  fi
else
  echo " <W> path exists => using already installed openmpi"
fi
export OMDIR=${OMIDIR}
cd ${HDIR}


# create XML file fro SCRAM
if [ "${FLGXMLFL}" = "TRUE" ]; then
  xmlfile="openmpi_"${OMVER}".xml"
  echo " <I>"
  echo " <I> creating openmpi tool definition XML file"
  if [ -e ${xmlfile} ]; then rm ${xmlfile}; fi; touch ${xmlfile}
  echo "  <tool name=\"openmpi\" version=\""${OMVER}"\">" >> ${xmlfile}
  tmppath=`find ${OMDIR} -type f -name libopenmpi.so\*`
#
	echo "OMDIR: "$OMDIR
	echo "TMPPATH: "$tmppath
#
  tmpcnt=`echo ${tmppath} | grep -o "/" | grep -c "/"`
  tmppath=`echo ${tmppath} | cut -f 1-${tmpcnt} -d "/"`
  for LIB in `cd ${tmppath}; ls *.so | cut -f 1 -d "." | sed -e 's/lib//'; cd ${HDIR}`; do
    echo "    <lib name=\""${LIB}"\"/>" >> ${xmlfile}
  done
  echo "    <client>" >> ${xmlfile}
  echo "      <Environment name=\"OM_BASE\" value=\""${OMDIR}"\"/>" >> ${xmlfile}
  echo "      <Environment name=\"LIBDIR\" default=\"\$OM_BASE/lib\"/>" >> ${xmlfile}
  echo "      <Environment name=\"INCLUDE\" default=\"\$OM_BASE/include\"/>" >> ${xmlfile}
  echo "    </client>" >> ${xmlfile}
  echo "  </tool>" >> ${xmlfile}
  if [ ! "$PWD" = "${HDIR}" ]; then
    mv ${xmlfile} ${HDIR}/
  fi

  if [ ! "$CMSSW_BASE" = "" ]; then
    cd $CMSSW_BASE
    tmpom=`scramv1 tool info openmpi | grep "OM_BASE" | cut -f2 -d"="`
    tmpxml=`find $CMSSW_BASE/config -type f -name qd.xml -printf %h`
    echo " <I>"
    echo " <I> openmpi version currently being used: "${tmpom}
    echo " <I> ...defined in "${tmpxml}
    cd ${tmpxml}; tmpxml=$PWD; cd ${HDIR}
    echo " <I>"
    echo " <I> If you want to override this version with the freshly produced "${xmlfile}","
    echo " <I> ...please type the following commands:"
    echo " <I>"
    echo "       cd $CMSSW_BASE"
    echo "       scramv1 tool remove openmpi"
    echo "       cp ${HDIR}/${xmlfile} ${tmpxml}/"
    echo "       scramv1 setup openmpi"
    echo "       cd -"
    echo " <I>"
  fi

fi


echo " -> openmpi installation directory is: " &&\
echo "  "${OMIDIR} && \
echo "" &&\
echo "   + please make sure to add (prepend)" &&\
echo "       "${OMIDIR}"/lib" &&\
echo "      to LD_LIBRARY_PATH environment variable" &&\
echo "   + please make sure to add (prepend)" &&\
echo "       "${OMIDIR}"/bin" &&\
echo "      to the PATH environment variable" &&\
echo "   + please make sure to create a configuration file '.mpd.conf'" &&\
echo "      in your home directory which at least should contain the line" &&\
echo "       MPD_SECRETWORD=(some 8 character long code)" &&\
echo "      and then to launch 'mpd' in the background before starting:" &&\
echo "       mpd &" &&\
echo "" &&\
echo ""


## update LD_LIBRARY_PATH
## setup 'mpd' -> configuration file...
## modify PATH to find current 'mpiexec'/'mpirun' first
