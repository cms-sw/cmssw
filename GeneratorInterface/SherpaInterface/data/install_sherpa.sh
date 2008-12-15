#!/bin/bash
#
#  file:        install_sherpa.sh
#  description: BASH script for the installation of the SHERPA MC generator,
#               downloads and installs SHERPA, HepMC2 and LHAPDF (if required),
#               can be used standalone or called from other scripts,
#  uses:        install_hepmc2.sh
#               install_lhapdf.sh
#               SHERPA patches/fixes [see below]
#
#  author:      Markus Merschmeyer, RWTH Aachen
#  date:        2008/12/09
#  version:     2.4
#

print_help() {
    echo "" && \
    echo "install_sherpa version 2.4" && echo && \
    echo "options: -v  version    define SHERPA version ( "${SHERPAVER}" )" && \
    echo "         -d  path       define (SHERPA) installation directory" && \
    echo "                         -> ( "${IDIR}" )" && \
    echo "         -f             require flags for 32-bit compilation ( "${FLAGS}" )" && \
    echo "         -F             apply inofficial fixes (LHAPDF,...)  ( "${FIXES}" )" && \
    echo "                         from this path ( "${FDIR}" )" && \
    echo "         -p  path       apply official SHERPA patches ( "${PATCHES}" )" && \
    echo "                         from this path ( "${PDIR}" )" && \
    echo "         -m  version    request HepMC2 installation ( "${HEPMC}", "${HVER}" )" && \
    echo "         -M  options    special HepMC2 options ( "${OPTHEPMC}" )" && \
    echo "         -l  version    request LHAPDF installation ( "${LHAPDF}", "${LVER}" )" && \
    echo "         -L  options    special LHAPDF options ( "${OPTLHAPDF}" )" && \
    echo "         -P             link (softlink) LHAPDF sets ( "${LINKPDF}" )" && \
    echo "                         or do a hardcopy if not set" && \
    echo "         -T             enable multithreading [V >= 1.1.0] ( "${MULTITHR}" )" && \
    echo "         -W  location   (web)location of SHERPA tarball ( "${SHERPAWEBLOCATION}" )" && \
    echo "         -S  filename   file name of SHERPA tarball ( "${SHERPAFILE}" )" && \
    echo "         -C  level      cleaning level of SHERPA installation ("${LVLCLEAN}" )" && \
    echo "                         -> 0: nothing, 1: +objects, 2: +sourcecode" && \
    echo "         -D             debug flag, compile with '-g' option ("${FLGDEBUG}" )" && \
    echo "         -I             installation flag ( "${FLGINSTL}" )" && \
    echo "                         -> use 'configure/make/make install' instead" && \
    echo "                         -> of 'TOOLS/makeinstall'" && \
    echo "         -h             display this help and exit" && echo
}


# save current path
HDIR=`pwd`


# dummy setup (if all options are missing)
IDIR="/tmp"                # installation directory
SHERPAVER="1.1.2"          # SHERPA version to be installed
SHCFLAGS=" "               # SHERPA compiler flags
SHMFLAGS=" -t"             # SHERPA 'make' flags
HEPMC="FALSE"              # install HepMC2
HVER="2.04.00"             # HepMC2 version  to be installed
OPTHEPMC=""                # special HepMC2 installation options
LHAPDF="FALSE"             # install LHAPDF
OPTLHAPDF=""               # special LHAPDF installation options
LINKPDF="FALSE"            # link (softlink) LHAPDF sets
LVER="5.6.0"               # LHAPDF version  to be installed
FLAGS="FALSE"              # apply SHERPA compiler/'make' flags
FIXES="FALSE"              # apply inofficial fixes
FDIR="./"                  # path containing fixes
PATCHES="FALSE"            # apply SHERPA patches
PDIR="./"                  # path containing patches
MULTITHR="FALSE"           # use multithreading
SHERPAWEBLOCATION=""       # (web)location of SHERPA tarball
SHERPAFILE=""              # file name of SHERPA tarball
LVLCLEAN=0                 # cleaning level (0-2)
FLGDEBUG="FALSE"           # debug flag for compilation
FLGINSTL="FALSE"           # installation flag


# get & evaluate options
while getopts :v:d:m:l:p:F:W:S:C:M:L:fPTDIh OPT
do
  case $OPT in
  v) SHERPAVER=$OPTARG ;;
  d) IDIR=$OPTARG ;;
  f) FLAGS=TRUE ;;
  F) FIXES=TRUE && FDIR=$OPTARG ;;
  m) HEPMC=TRUE && HVER=$OPTARG ;;
  M) OPTHEPMC=$OPTARG ;;
  l) LHAPDF=TRUE && LVER=$OPTARG ;;
  L) OPTLHAPDF=$OPTARG ;;
  P) LINKPDF=TRUE ;;
  p) PATCHES=TRUE && PDIR=$OPTARG ;;
  T) MULTITHR=TRUE ;;
  W) SHERPAWEBLOCATION=$OPTARG ;;
  S) SHERPAFILE=$OPTARG ;;
  C) LVLCLEAN=$OPTARG ;;
  D) FLGDEBUG=TRUE ;;
  I) FLGINSTL=TRUE ;;
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
shpatchfile="sherpa_patches_"${SHERPAVER}".tgz" # official patches for current SHERPA version
shfixfile="sherpa_fixes_"${SHERPAVER}".tgz"     # fixes for current SHERPA version
shshifile="install_sherpa.sh"        # this script
shhmifile="install_hepmc2.sh"        # script for HepMC2 installation
shlhifile="install_lhapdf.sh"        # script for LHAPDF installation


# set SHERPA (HepMC2,LHAPDF) download location
if [ "$SHERPAWEBLOCATION" = "" ]; then
  SHERPAWEBLOCATION="http://www.hepforge.org/archive/sherpa"
  FLOC=" "
else
  FLOC=" -W "${SHERPAWEBLOCATION}
fi
if [ "$SHERPAFILE" = "" ]; then
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
cd ${FDIR}; FDIR=`pwd`; cd ${HDIR}

echo " SHERPA (HepMC2,LHAPDF) installation: "
echo "  -> SHERPA version: '"${SHERPAVER}"' ("${va}","${vb}","${vc}")"
echo "  -> installation directory: '"${IDIR}"'"
echo "  -> SHERPA patches: '"${PATCHES}"' in '"${PDIR}"'"
echo "  -> SHERPA fixes: '"${FIXES}"' in '"${FDIR}"'"
echo "  -> flags: '"${FLAGS}"'"
echo "  -> multithreading: '"${MULTITHR}"'"
echo "  -> SHERPA location: '"${SHERPAWEBLOCATION}"'"
echo "  -> SHERPA file name: '"${SHERPAFILE}"'"
echo "  -> cleaning level: '"${LVLCLEAN}"'"
echo "  -> debugging mode: '"${FLGDEBUG}"'"
echo "  -> use conf/make/make: '"${FLGINSTL}"'"
echo "  -> HepMC2: '"${HEPMC}"', version '"${HVER}"'"
echo "  ->   options: "${OPTHEPMC}
echo "  -> LHAPDF: '"${LHAPDF}"', version '"${LVER}"'"
echo "  ->   options: "${OPTLHAPDF}
echo "  -> link PDFsets: '"${LINKPDF}"'"


# forward flags to HepMC2/LHAPDF
IFLG=" "
if [ "$FLAGS" = "TRUE" ]; then
  IFLG=" -f"
fi
if [ ${LVLCLEAN} -gt 0 ]; then
  IFLG=${IFLG}" -C "${LVLCLEAN}
fi
if [ "$FLGDEBUG" = "TRUE" ]; then
  IFLG=${IFLG}" -D"
fi


# set path to local SHERPA installation
export SHERPADIR=${IDIR}"/SHERPA-MC-"${SHERPAVER}
export SHERPAIDIR=${IDIR}"/SHERPA_"${SHERPAVER}


# check HepMC2 installation (if required)
if [ "$HEPMC" = "TRUE" ]; then
  OPTHEPMC=${OPTHEPMC}" -v "${HVER}" -d "${IDIR}
  if [ ! "$HEPMC2DIR" = "" ]; then
    echo " -> HepMC2 directory is: "${HEPMC2DIR}
    if [ ! -e ${HEPMC2DIR} ]; then
      echo " -> ... and does not exist: installing HepMC2..."
      echo " -> ... with options: "${OPTHEPMC}
      ${MSI}/${shhmifile} ${IFLG} ${FLOC} ${OPTHEPMC}
    else
      echo " -> ... and exists: Installation cancelled!"
    fi
  else
    export HEPMC2DIR=${IDIR}"/HepMC-"${HVER}
    export HEPMC2IDIR=${IDIR}"/HEPMC_"${HVER}
    echo " -> no HepMC2 directory specified, trying installation"
    echo "     into "${HEPMC2IDIR}
    ${MSI}/${shhmifile} ${IFLG} ${FLOC} ${OPTHEPMC}
  fi
###FIXME
  if [ "${SHERPAVER}" = "1.0.11" ]; then
    SHCFLAGS=${SHCFLAGS}" --enable-hepmc2"
    SHMFLAGS=${SHMFLAGS}" --copt --enable-hepmc2"
  else
    SHCFLAGS=${SHCFLAGS}" --enable-hepmc2="${HEPMC2IDIR}
    SHMFLAGS=${SHMFLAGS}" --copt --enable-hepmc2="${HEPMC2IDIR}
  fi
  export HEPMC2DIR=${HEPMC2IDIR}
###FIXME
fi


# check LHAPDF installation (if required)
PATCHLHAPDF=FALSE
FIXLHAPDF=FALSE
if [ "$LHAPDF" = "TRUE" ]; then
  OPTLHAPDF=${OPTLHAPDF}" -v "${LVER}" -d "${IDIR}
  if [ ! "$LHAPDFDIR" = "" ]; then
    echo " -> LHAPDF directory is: "${LHAPDFDIR}
    if [ ! -e ${LHAPDFDIR} ]; then
      echo " -> ... and does not exist: installing LHAPDF..."
      echo " -> ... with options: "${OPTLHAPDF}
      ${MSI}/${shlhifile} ${IFLG} ${FLOC} ${OPTLHAPDF}
    else
      echo " -> ... and exists: Installation cancelled!"
    fi
  else
    export LHAPDFDIR=${IDIR}"/lhapdf-"${LVER}
    export LHAPDFIDIR=${IDIR}"/LHAPDF_"${LVER}
    echo " -> no LHAPDF directory specified, trying installation"
    echo "     into "${LHAPDFIDIR}
    ${MSI}/${shlhifile} ${IFLG} ${FLOC} ${OPTLHAPDF}
  fi
###FIXME
  if [ "${SHERPAVER}" = "1.0.11" ]; then
    SHCFLAGS=${SHCFLAGS}" --enable-lhapdf"
    SHMFLAGS=${SHMFLAGS}" --copt --enable-lhapdf"
  else
    SHCFLAGS=${SHCFLAGS}" --enable-lhapdf="${LHAPDFIDIR}
    SHMFLAGS=${SHMFLAGS}" --copt --enable-lhapdf="${LHAPDFIDIR}
  fi
###FIXME
  PATCHLHAPDF=TRUE
  FIXLHAPDF=TRUE
  export LHAPDFDIR=${LHAPDFIDIR}
fi
#echo "STOP"
#sleep 1000

# download and extract SHERPA
cd ${IDIR}
if [ ! -d ${SHERPADIR} ]; then
  if [ `echo ${SHERPAWEBLOCATION} | grep -c "http:"` -gt 0 ]; then
    echo " -> downloading SHERPA "${SHERPAVER}" from "${SHERPAWEBLOCATION}/${SHERPAFILE}
    wget ${SHERPAWEBLOCATION}/${SHERPAFILE}
  elif [ `echo ${SHERPAWEBLOCATION} | grep -c "srm:"` -gt 0 ]; then
    echo " -> srm-copying SHERPA "${SHERPAVER}" from "${SHERPAWEBLOCATION}/${SHERPAFILE}
    srmcp ${SHERPAWEBLOCATION}/${SHERPAFILE} file:////${SHERPAFILE}
  else
    echo " -> copying SHERPA "${SHERPAVER}" from "${SHERPAWEBLOCATION}/${SHERPAFILE}
    cp ${SHERPAWEBLOCATION}/${SHERPAFILE} ./
  fi
  tar -xzf ${SHERPAFILE}
  rm ${SHERPAFILE}
else
  echo " <W> path exists => using already installed SHERPA"
  echo " <W>  -> this might cause problems with some fixes and/or patches !!!"
fi
cd ${HDIR}


# add compiler & linker flags
M_CFLAGS=""
M_FFLAGS=""
M_CXXFLAGS=""
M_LDFLAGS=""
M_FFLGS2=""
M_CXXFLGS2=""
CF32BIT=""
if [ "$FLAGS" = "TRUE" ]; then
  CF32BIT="-m32"
  M_CFLAGS="CFLAGS="${CF32BIT}
  M_FFLAGS="FFLAGS="${CF32BIT}
  M_CXXFLAGS="CXXFLAGS="${CF32BIT}
  M_LDFLAGS="LDFLAGS="${CF32BIT}
  M_FFLGS2=${M_FFLGS2}" --f "${CF32BIT}
  M_CXXFLGS2=${M_CXXFLGS2}" --cxx "${CF32BIT}
fi
CFDEBUG=""
if [ "$FLGDEBUG" = "TRUE" ]; then
  CFDEBUG="-g"
  M_FFLGS2=${M_FFLGS2}" --f "${CFDEBUG}
  M_CXXFLGS2=${M_CXXFLGS2}" --cxx "${CFDEBUG}
fi
CONFFLG=${M_CFLAGS}" "${M_FFLAGS}" "${M_CXXFLAGS}" "${M_LDFLAGS}
if [ "$FLAGS" = "TRUE" ]; then
  M_CFLAGS=" --copt "${M_CFLAGS}
  M_LDFLAGS=" --copt "${M_LDFLAGS}
fi
MAKEFLG=${M_CFLAGS}" "${M_LDFLAGS}" "${M_CXXFLGS2}" "${M_FFLGS2}
SHCFLAGS=${SHCFLAGS}" "${CONFFLG}
SHMFLAGS=${SHMFLAGS}" "${MAKEFLG}

if [ ${vb} -ge 1 ]; then  # enable multithreading if version >= 1.1.X
  if [ "$MULTITHR" = "TRUE" ]; then
    SHCFLAGS=${SHCFLAGS}" --enable-multithread"
    SHMFLAGS=${SHMFLAGS}" --copt --enable-multithread"
  fi
fi


# apply the necessary patches
cd ${SHERPADIR}
if [ "$PATCHES" = "TRUE" ]; then
  echo " <I> applying patches to SHERPA..."
  if [ -e ${PDIR}/${shpatchfile} ]; then
    pfilelist=`tar -xzvf ${PDIR}/${shpatchfile}`
    for pfile in `echo ${pfilelist}`; do
      echo "  -> applying patch: "${pfile}
      patch -p0 < ${pfile}
      echo " <I> (patches) removing file "${pfile}
      rm ${pfile}
    done
  else
    echo " <W> file "${PDIR}/${shpatchfile}" does not exist,"
    echo " <W>  cannot apply Sherpa patches"
  fi
fi
cd ${HDIR}


# apply the necessary fixes
cd ${SHERPADIR}
if [ "$FIXES" = "TRUE" ]; then
  echo " <I> applying fixes to SHERPA..."
  if [ -e ${FDIR}/${shfixfile} ]; then
    ffilelist=`tar -xzvf ${FDIR}/${shfixfile}`
    for ffile in `echo ${ffilelist}`; do
      echo "  -> applying fix: "${ffile}
      if [ `echo ${ffile} | grep -i -c lhapdf` -gt 0 ]; then
        ./${ffile} ${LHAPDFDIR}
      else
        ./${ffile} ${PWD}
      fi
      echo " <I> (fixes) removing file "${ffile}
      rm ${ffile}
    done
  else
    echo " <W> file "${FDIR}/${shfixfile}" does not exist,"
    echo " <W>  cannot apply Sherpa fixes"
  fi
fi
cd ${HDIR}



# compile and install SHERPA
cd ${SHERPADIR}
if [ -e ${SHERPADIR}/bin/Sherpa ]; then
  echo " <W> installed SHERPA exists, cleaning up"
  ./TOOLS/makeinstall --clean-up
fi

if [ "$FLGINSTL" = "TRUE" ]; then
  if [ "$FIXES" = "TRUE" ]; then
    aclocal
    autoheader
    automake
    autoconf
  fi
#  echo " -> configuring SHERPA with flags: --prefix="${SHERPADIR}" "${SHCFLAGS}
#  echo "./configure --prefix="${SHERPADIR}" "${SHCFLAGS} > ../sherpa_configr.cmd
#  ./configure --prefix=${SHERPADIR} ${SHCFLAGS} > ../sherpa_install.log 2>&1
  echo " -> configuring SHERPA with flags: --prefix="${SHERPAIDIR}" "${SHCFLAGS}
  echo "./configure --prefix="${SHERPAIDIR}" "${SHCFLAGS} > ../sherpa_configr.cmd
  ./configure --prefix=${SHERPAIDIR} ${SHCFLAGS} > ../sherpa_install.log 2>&1
  echo "-> making SHERPA with flags: "${CONFFLG}
  echo "make "${CONFFLG} >> ../sherpa_configr.cmd
  make ${CONFFLG} >> ../sherpa_install.log 2>&1
  echo "-> making install SHERPA with flags: "${CONFFLG}
  echo "make install "${CONFFLG} >> ../sherpa_configr.cmd
  make install ${CONFFLG} >> ../sherpa_install.log 2>&1
  cd ${HDIR}
  rm -rf ${SHERPADIR}
else
  echo " -> installing SHERPA with flags: "${SHMFLAGS}
  echo "./TOOLS/makeinstall "${SHMFLAGS} > ../sherpa_install.cmd
  ./TOOLS/makeinstall ${SHMFLAGS}
  if [ ${LVLCLEAN} -gt 0 ]; then 
    echo " -> cleaning up SHERPA installation, level: "${LVLCLEAN}" ..."
    if [ ${LVLCLEAN} -ge 1 ]; then  # normal cleanup (objects)
      ./TOOLS/makeinstall --clean-up
    fi
    if [ ${LVLCLEAN} -ge 2 ]; then  # clean also sourcecode
      find ./ -type f -name 'Makefile*' -exec rm -rf {} \;
      find ./ -type d -name '.deps'     -exec rm -rf {} \;
      rm -rf autom4te.cache
      rm stamp-h1 missing ltmain.sh libtool install-sh depcomp config* aclocal.m4 acinclude.m4
      rm -rf AHADIC++ AMEGIC++ AMISIC++ ANALYSIS APACIC++ ATOOLS BEAM
      rm -rf EXTRA_XS HADRONS++ HELICITIES MODEL PDF PHASIC++ PHOTONS++ TOOLS
      rm AUTHORS COPYING README
      rm ChangeLog INSTALL NEWS
    fi
  fi
  cd ${HDIR}
fi
export SHERPADIR=${SHERPAIDIR}
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
#    cp ${SHERPADIR}/share/SHERPA-MC/makelibs ${SHERPADIR}/${shdir}/Run/makelibs
### FIXME
fi


# get LHAPDFs into SHERPA... (now with symbolic links)
if [ "$LHAPDF" = "TRUE" ]; then
  pdfdir=`find ${LHAPDFDIR} -type d -name PDFsets`
  if [ ! -e ${pdfdir} ]; then
    echo " <E> PDFsets of LHAPDF not found, stopping..."
    exit 1
  fi
  if [ "${LINKPDF}" = "TRUE" ] && [ ! "${pdfdir}" = "" ]; then
    ln -s ${pdfdir} ${SHERPADIR}/share/SHERPA-MC/PDFsets
  else
    cp -r ${pdfdir} ${SHERPADIR}/share/SHERPA-MC/
  fi
fi


# summarize installation
echo " <I> Summary of the SHERPA installation:"
if [ "$HEPMC" = "TRUE" ]; then
echo " <I> HepMC2 version "${HEPMC2VER}" installed in "${HEPMC2DIR}
fi
if [ "$LHAPDF" = "TRUE" ]; then
echo " <I> LHAPDF version "${LHAPDFVER}" installed in "${LHAPDFDIR}
echo " <I>  -> before using SHERPA please define"
echo " <I>  -> export LHAPATH="${pdfdir}
fi
echo " <I> SHERPA version "${SHERPAVER}" installed in "${SHERPADIR}








