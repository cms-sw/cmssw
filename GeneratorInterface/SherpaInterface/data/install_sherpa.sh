#!/bin/bash
#
#  file:        install_sherpa.sh
#  description: BASH script for the installation of the SHERPA MC generator,
#               downloads and installs SHERPA, HepMC2 and LHAPDF (if required),
#               can be used standalone or called from other scripts,
#  uses:        install_hepmc2.sh
#               install_lhapdf.sh
#               SHERPA patches [see below]
#
#  author:      Markus Merschmeyer, RWTH Aachen
#  date:        2008/05/13
#  version:     1.5
#

print_help() {
    echo "" && \
    echo "install_sherpa version 1.5" && echo && \
    echo "options: -v  version    define SHERPA version ( "${SHERPAVER}" )" && \
    echo "         -d  path       define SHERPA installation directory" && \
    echo "                         -> ( "${IDIR}" )" && \
    echo "         -f             require flags for 32-bit compilation ( "${FLAGS}" )" && \
    echo "         -F             apply inofficial fixes (LHAPDF,...)  ( "${FIXES}" )" && \
    echo "         -m  version    request HepMC2 installation ( "${HEPMC}", "${HVER}" )" && \
    echo "         -l  version    request LHAPDF installation ( "${LHAPDF}", "${LVER}" )" && \
    echo "         -p  path       apply patches from this path" && \
    echo "                         -> ( "${PDIR}" )" && \
    echo "         -M             enable multithreading [V >= 1.1.0] ( "${MULTITHR}" )" && \
    echo "         -W  location   (web)location of SHERPA tarball ( "${SHERPAWEBLOCATION}" )" && \
    echo "         -S  filename   file name of SHERPA tarball ( "${SHERPAFILE}" )" && \
    echo "         -h             display this help and exit" && echo
}


# save current path
HDIR=`pwd`

# dummy setup (if all options are missing)
IDIR="/tmp"                # installation directory
SHERPAVER="1.1.1"          # SHERPA version to be installed
SHCFLAGS=" "               # SHERPA compiler flags
SHMFLAGS=" -t"             # SHERPA 'make' flags
HEPMC="FALSE"              # install HepMC2
HVER="2.01.10"             # HepMC2 version  to be installed
LHAPDF="FALSE"             # install LHAPDF
LVER="5.3.1"               # LHAPDF version  to be installed
FLAGS="FALSE"              # apply SHERPA compiler/'make' flags
FIXES="FALSE"              # apply inofficial fixes
PATCHES="FALSE"            # apply SHERPA patches
PDIR="./"                  # path containing patches
MULTITHR="FALSE"           # use multithreading
SHERPAWEBLOCATION=" "      # (web)location of SHERPA tarball
SHERPAFILE=" "             # file name of SHERPA tarball


# get & evaluate options
while getopts :v:d:m:l:p:W:S:fFMh OPT
do
  case $OPT in
  v) SHERPAVER=$OPTARG ;;
  d) IDIR=$OPTARG ;;
  f) FLAGS=TRUE ;;
  F) FIXES=TRUE ;;
  m) HEPMC=TRUE && HVER=$OPTARG ;;
  l) LHAPDF=TRUE && LVER=$OPTARG ;;
  p) PATCHES=TRUE && PDIR=$OPTARG ;;
  M) MULTITHR=TRUE ;;
  W) SHERPAWEBLOCATION=$OPTARG ;;
  S) SHERPAFILE=$OPTARG ;;
  h) print_help && exit 0 ;;
  \?)
    shift `expr $OPTIND - 1`
    if [ "$1" = "--help" ]; then print_help && exit 0;
    else 
      echo -n "install_sherpa: error: unrecognized option "
      if [ $OPTARG != "-" ]; then echo "'-$OPTARG'. try '-h'"
      else echo "'$1'. try '-h'"
      fi
      print_help && exit 1
    fi
  esac
done


# set up file names
MSI=$HDIR                            # main installation directory
###shpatchfile="sherpa_patches.tgz"     # patches for current SHERPA version
shpatchfile="sherpa_patches_"${SHERPAVER}".tgz"     # patches for current SHERPA version
shshifile="install_sherpa.sh"        # this script
shhmifile="install_hepmc2.sh"        # script for HepMC2 installation
shlhifile="install_lhapdf.sh"        # script for LHAPDF installation


# set SHERPA download location
if [ "$SHERPAWEBLOCATION" = " " ]; then
  SHERPAWEBLOCATION="http://www.hepforge.org/archive/sherpa"
fi
if [ "$SHERPAFILE" = " " ]; then
  SHERPAFILE="Sherpa-"${SHERPAVER}".tar.gz"
fi


# analyze SHERPA version
va=`echo ${SHERPAVER} | cut -f1 -d"."`
vb=`echo ${SHERPAVER} | cut -f2 -d"."`
vc=`echo ${SHERPAVER} | cut -f3 -d"."`

# make SHERPA version a global variable
export SHERPAVER=${SHERPAVER}
export HEPMC2VER=${HVER}
export LHAPDFVER=${LVER}
# always use absolute path names...
cd ${IDIR}; IDIR=`pwd`; cd ${HDIR}
cd ${PDIR}; PDIR=`pwd`; cd ${HDIR}

echo " SHERPA (HepMC2,LHAPDF) installation: "
echo "  -> SHERPA version: '"${SHERPAVER}"' ("${va}","${vb}","${vc}")"
echo "  -> installation directory: '"${IDIR}"'"
echo "  -> SHERPA patches: '"${PATCHES}"' in '"${PDIR}"'"
echo "  -> flags: '"${FLAGS}"'"
echo "  -> fixes: '"${FIXES}"'"
echo "  -> multithreading: '"${MULTITHR}"'"
echo "  -> SHERPA location: '"${SHERPAWEBLOCATION}"'"
echo "  -> SHERPA file name: '"${SHERPAFILE}"'"
echo "  -> HepMC2: '"${HEPMC}"', version '"${HVER}"'"
echo "  -> LHAPDF: '"${LHAPDF}"', version '"${LVER}"'"


# forward flags to HepMC2/LHAPDF
IFLG=" "
if [ "$FLAGS" = "TRUE" ]; then
  IFLG=" -f"
fi


# set path to local SHERPA installation
export SHERPADIR=${IDIR}"/SHERPA-MC-"${SHERPAVER}


# check HepMC2 installation (if required)
if [ "$HEPMC" = "TRUE" ]; then
  if [ ! "$HEPMC2DIR" = "" ]; then
    echo " -> HepMC2 directory is: "${HEPMC2DIR}
    if [ ! -e ${HEPMC2DIR} ]; then
      echo " -> ... and does not exist: installing HepMC2..."
      ${MSI}/${shhmifile} -v ${HVER} -d ${IDIR} ${IFLG}
    else
      echo " -> ... and exists: Installation cancelled!"
    fi
  else
    export HEPMC2DIR=${IDIR}"/HepMC-"${HVER}
    echo " -> no HepMC2 directory specified, trying installation"
    echo "     into "${HEPMC2DIR}
    ${MSI}/${shhmifile} -v ${HVER} -d ${IDIR} ${IFLG}
  fi
###FIXME
if [ "${SHERPAVER}" = "1.0.11" ]; then
  SHCFLAGS=${SHCFLAGS}" --enable-hepmc2"
  SHMFLAGS=${SHMFLAGS}" --copt --enable-hepmc2"
elif [ "${SHERPAVER}" = "1.1.0" ]; then
  SHCFLAGS=${SHCFLAGS}" --enable-hepmc2="${HEPMC2DIR}
  SHMFLAGS=${SHMFLAGS}" --copt --enable-hepmc2="${HEPMC2DIR}
elif [ "${SHERPAVER}" = "1.1.1" ]; then
  SHCFLAGS=${SHCFLAGS}" --enable-hepmc2="${HEPMC2DIR}
  SHMFLAGS=${SHMFLAGS}" --copt --enable-hepmc2="${HEPMC2DIR}
else
  SHCFLAGS=${SHCFLAGS}" --enable-hepmc2="${HEPMC2DIR}
  SHMFLAGS=${SHMFLAGS}" --copt --enable-hepmc2="${HEPMC2DIR}
fi
###FIXME
fi


# check LHAPDF installation (if required)
PATCHLHAPDF=FALSE
if [ "$LHAPDF" = "TRUE" ]; then
  if [ ! "$LHAPDFDIR" = "" ]; then
    echo " -> LHAPDF directory is: "${LHAPDFDIR}
    if [ ! -e ${LHAPDFDIR} ]; then
      echo " -> ... and does not exist: installing LHAPDF..."
      ${MSI}/${shlhifile} -v ${LVER} -d ${IDIR} ${IFLG}
    else
      echo " -> ... and exists: Installation cancelled!"
    fi
  else
    export LHAPDFDIR=${IDIR}"/lhapdf-"${LVER}
    echo " -> no LHAPDF directory specified, trying installation"
    echo "     into "${LHAPDFDIR}
    ${MSI}/${shlhifile} -v ${LVER} -d ${IDIR} ${IFLG}
  fi
###FIXME
if [ "${SHERPAVER}" = "1.0.11" ]; then
  SHCFLAGS=${SHCFLAGS}" --enable-lhapdf"
  SHMFLAGS=${SHMFLAGS}" --copt --enable-lhapdf"
elif [ "${SHERPAVER}" = "1.1.0" ]; then
  SHCFLAGS=${SHCFLAGS}" --enable-lhapdf="${LHAPDFDIR}
  SHMFLAGS=${SHMFLAGS}" --copt --enable-lhapdf="${LHAPDFDIR}
elif [ "${SHERPAVER}" = "1.1.1" ]; then
  SHCFLAGS=${SHCFLAGS}" --enable-lhapdf="${LHAPDFDIR}
  SHMFLAGS=${SHMFLAGS}" --copt --enable-lhapdf="${LHAPDFDIR}
else
  SHCFLAGS=${SHCFLAGS}" --enable-lhapdf="${LHAPDFDIR}
  SHMFLAGS=${SHMFLAGS}" --copt --enable-lhapdf="${LHAPDFDIR}
fi
###FIXME
  PATCHLHAPDF=TRUE
fi


# download and extract SHERPA
cd ${IDIR}
if [ ! -d ${SHERPADIR} ]; then
  echo " -> downloading SHERPA "${SHERPAVER}" from "${SHERPAWEBLOCATION}/${SHERPAFILE}
  wget ${SHERPAWEBLOCATION}/${SHERPAFILE}
  tar -xzf ${SHERPAFILE}
  rm ${SHERPAFILE}
else
  echo " <W> path exists => using already installed SHERPA"
  echo " <W>  -> this might cause problems with some fixes and/or patches !!!"
fi
cd ${HDIR}


# add compiler & linker flags
if [ "$FLAGS" = "TRUE" ]; then
### FIXME (use CMSSW script to make gcc 32-bit compatible?)
#  SHCFLAGS=${SHCFLAGS}" CFLAGS=-m32 FFLAGS=-m32 CXXFLAGS=-m32 LDFLAGS=-m32"
#  SHMFLAGS=${SHMFLAGS}" --copt CFLAGS=-m32 --copt FFLAGS=-m32 --copt CXXFLAGS=-m32 --copt LDFLAGS=-m32"
### FIXME
# NEW???
  SHCFLAGS=${SHCFLAGS}" CXXFLAGS=-m32 FFLAGS=-m32 CFLAGS=-m32 LDFLAGS=-m32"
  SHMFLAGS=${SHMFLAGS}" --cxx -m32 --f -m32 --copt CFLAGS=-m32 --copt LDFLAGS=-m32"
### FIXME
##  if [ "$SHERPAVER" = "1.1.0" ]; then
  if [ ${vb} -ge 1 ]; then  # version >= 1.1.X
    if [ "$MULTITHR" = "TRUE" ]; then
      SHCFLAGS=${SHCFLAGS}" --enable-multithread"
      SHMFLAGS=${SHMFLAGS}" --copt --enable-multithread"
    fi
  fi
fi


# apply the necessary patches & fixes
cd ${SHERPADIR}
if [ "$PATCHES" = "TRUE" ]; then
  echo " <I> applying patches and fixes to SHERPA..."
  if [ -e ${PDIR}/${shpatchfile} ]; then
    cd ${PDIR}
    tar -xzvf ${shpatchfile}
#    rm ${shpatchfile}
    cd -
  fi
  fcmd1a="ls ${PDIR}/*.patch"
  fcmd1b="fgrep -i -v -e lha"
  for pfile1 in `${fcmd1a} | ${fcmd1b}`; do
    echo "  -> applying patch: "${pfile1}
    patch -p0 < ${pfile1}
  done
  if [ "$PATCHLHAPDF" = "TRUE" ]; then
    fcmd2="ls ${PDIR}/*lha*.patch"
    for pfile2 in `${fcmd2}`; do
      echo "  -> applying patch: "${pfile2}
      patch -p0 < ${pfile2}
    done
  fi
  rm ${PDIR}/*.patch
  if [ "$SHERPAVER" = "1.0.11" ]; then
### FIXME (probably not needed in later SHERPA versions)
    echo " <W> fixing AMEGIC++-2.0.11/Model/Makefile.am"
    sed -e 's/CXXFLAGS/AM_CXXFLAGS/' < AMEGIC++-2.0.11/Model/Makefile.am > AMEGIC++-2.0.11/Model/Makefile.am.tmp
    mv AMEGIC++-2.0.11/Model/Makefile.am.tmp AMEGIC++-2.0.11/Model/Makefile.am
### FIXME
### FIXME
    if [ -e ${PDIR}/LesHouches_Interface.C_MM_SS ]; then
      echo " <W> fixing MODEL-1.0.11/Main/LesHouches_Interface.C"
      mv ${SHERPADIR}/MODEL-1.0.11/Main/LesHouches_Interface.C ${SHERPADIR}/MODEL-1.0.11/Main/LesHouches_Interface.C_orig
      cp ${PDIR}/LesHouches_Interface.C_MM_SS ${SHERPADIR}/MODEL-1.0.11/Main/LesHouches_Interface.C
    fi
### FIXME
  elif [ "$SHERPAVER" = "1.1.0" ]; then
### FIXME
    echo " <W> fixing MODEL/Interaction_Models/Makefile.am"
    sed -e 's/CXXFLAGS/AM_CXXFLAGS/' < MODEL/Interaction_Models/Makefile.am > MODEL/Interaction_Models/Makefile.am.tmp
    mv MODEL/Interaction_Models/Makefile.am.tmp MODEL/Interaction_Models/Makefile.am
### FIXME
  fi
  if [ "$FIXES" = "TRUE" ]; then
### FIXME
    echo " <W> fixing "${SHERPADIR}"/acinclude.m4 : libLHAPDF.a -> liblhapdf.a, ..."
    sed -e 's:lib/libLHAPDF.a:lib/archive/liblhapdf.a '${LHAPDFDIR}'/lib/archive/liblhapdf_dummy.a:' < ${SHERPADIR}/acinclude.m4 > ${SHERPADIR}/acinclude.m4.tmp
    mv  ${SHERPADIR}/acinclude.m4.tmp ${SHERPADIR}/acinclude.m4
### FIXME
  fi
fi
cd ${HDIR}


# compile and install SHERPA
cd ${SHERPADIR}
if [ -e ${SHERPADIR}/bin/Sherpa ]; then
  echo " <W> installed SHERPA exists, cleaning up"
  ./TOOLS/makeinstall --clean-up
fi
###  ./TOOLS/makeinstall --clean-up
#MM#echo " -> configuring SHERPA with flags: "${SHCFLAGS}
#MM#./configure --prefix=${SHERPADIR} ${SHCFLAGS}
echo " -> installing SHERPA with flags: "${SHMFLAGS}
./TOOLS/makeinstall ${SHMFLAGS}
if [ ! -e ./bin/Sherpa ]; then
  echo " <E> -------------------------------------------------------"
  echo " <E> -------------------------------------------------------"
  echo " <E> installation of SHERPA failed, dumping log file..."
  echo " <E> -------------------------------------------------------"
  cat sherpa_install.log
  echo " <E> -------------------------------------------------------"
  echo " <E> -------------------------------------------------------"
else
  echo " <I> installation of SHERPA was successful..."
fi
echo " -> cleaning up SHERPA installation..."
./TOOLS/makeinstall --clean-up
shdir=`ls | grep "SHERPA"`
echo " <I> SHERPA directory is: "${shdir}
cp ./bin/Sherpa ./${shdir}/Run/
cd ${HDIR}


# fix 'makelibs' script for 32-bit compatibility
if [ "$FLAGS" = "TRUE" ]; then
  echo " <W> setting 32bit flags in 'makelibs' script !!!"
### FIXME (use CMSSW script to make gcc 32-bit compatible?)
    CNFFLG="CFLAGS=-m32 FFLAGS=-m32 CXXFLAGS=-m32 LDFLAGS=-m32"
    MKEFLG="CFLAGS=-m32 FFLAGS=-m32 CXXFLAGS=\"-O2 -m32\" LDFLAGS=-m32"
    sed -e "s/configure/configure ${CNFFLG}/" < ${SHERPADIR}/share/SHERPA-MC/makelibs > ${SHERPADIR}/share/SHERPA-MC/makelibs.tmp
    rm ${SHERPADIR}/share/SHERPA-MC/makelibs
    sed -e "s/-j2 \"CXXFLAGS=-O2\"/-j2 ${MKEFLG}/" < ${SHERPADIR}/share/SHERPA-MC/makelibs.tmp > ${SHERPADIR}/share/SHERPA-MC/makelibs
    rm ${SHERPADIR}/share/SHERPA-MC/makelibs.tmp
    chmod 755 ${SHERPADIR}/share/SHERPA-MC/makelibs
    cp ${SHERPADIR}/share/SHERPA-MC/makelibs ${SHERPADIR}/${shdir}/Run/makelibs
### FIXME
fi


# get LHAPDFs into SHERPA... (now with symbolic links)
if [ "$LHAPDF" = "TRUE" ]; then
  if [ -d ${LHAPDFDIR}/../PDFsets ]; then
#    cp -r ${LHAPDFDIR}/../PDFsets ${SHERPADIR}/share/SHERPA-MC/
    ln -s ${LHAPDFDIR}/../PDFsets ${SHERPADIR}/share/SHERPA-MC/PDFsets
  elif [ -d ${LHAPDFDIR}/PDFsets ]; then
#    cp -r ${LHAPDFDIR}/PDFsets ${SHERPADIR}/share/SHERPA-MC/
    ln -s ${LHAPDFDIR}/PDFsets ${SHERPADIR}/share/SHERPA-MC/PDFsets
  else
    echo " <E> PDFsets of LHAPDF not found"
  fi
fi
