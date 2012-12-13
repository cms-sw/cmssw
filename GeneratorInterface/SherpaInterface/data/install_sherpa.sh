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
#  date:        2011/02/28
#  version:     3.0
#

print_help() {
    echo "" && \
    echo "install_sherpa version 3.0" && echo && \
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
    echo "         -P  name       automatically retrieve LHAPDF set ( "${PDFSET}" )" && \
    echo "         -S             link (softlink) LHAPDF sets ( "${LINKPDF}" )" && \
    echo "                         or do a hardcopy if not set" && \
    echo "         -W  location   (web)location of SHERPA tarball ( "${SHERPAWEBLOCATION}" )" && \
    echo "         -Y  filename   file name of SHERPA tarball ( "${SHERPAFILE}" )" && \
    echo "         -C  level      cleaning level of SHERPA installation ("${LVLCLEAN}" )" && \
    echo "                         -> 0: nothing, 1: +objects, 2: +sourcecode" && \
    echo "         -T             enable multithreading [V >= 1.1.0]" && \
    echo "         -A             enable analysis [V >= 1.2.0]" && \
    echo "         -D             debug flag, compile with '-g' option ("${FLGDEBUG}" )" && \
    echo "         -I             installation flag ( "${FLGINSTL}" )" && \
    echo "                         -> use './TOOLS/makeinstall'" && \
    echo "                         -> instead of 'configure/make/make install'" && \
    echo "         -X             create XML file for tool override in CMSSW ( "${FLGXMLFL}" )" && \
    echo "         -Z             use multiple CPU cores if available ( "${FLGMCORE}" )" && \
    echo "         -K             keep SHERPA source code tree after installation ( "${FGLKEEPT}" )" && \
    echo "         -h             display this help and exit" && echo
}


# save current path
HDIR=`pwd`


# get absolute path of this script
SCRIPT=$(readlink -f $0)
#echo "XXX: "$SCRIPT
SCRIPTPATH=`dirname $SCRIPT`
#echo "YYY: "$SCRIPTPATH


# dummy setup (if all options are missing)
IDIR="/tmp"                # installation directory
SHERPAVER="1.2.0"          # SHERPA version to be installed
SHCFLAGS=" "               # SHERPA compiler flags
SHMFLAGS=" -t"             # SHERPA 'make' flags
HEPMC="FALSE"              # install HepMC2
HVER="2.05.00"             # HepMC2 version  to be installed
OPTHEPMC=""                # special HepMC2 installation options
LHAPDF="FALSE"             # install LHAPDF
OPTLHAPDF=""               # special LHAPDF installation options
PDFSET=""                  # name of PDFset to download
LINKPDF="FALSE"            # link (softlink) LHAPDF sets
LVER="5.6.0"               # LHAPDF version  to be installed
FLAGS="FALSE"              # apply SHERPA compiler/'make' flags
FIXES="FALSE"              # apply inofficial fixes
FDIR="./"                  # path containing fixes
PATCHES="FALSE"            # apply SHERPA patches
PDIR="./"                  # path containing patches
SHERPAWEBLOCATION=""       # (web)location of SHERPA tarball
SHERPAFILE=""              # file name of SHERPA tarball
LVLCLEAN=0                 # cleaning level (0-2)
FLGDEBUG="FALSE"           # debug flag for compilation
FLGINSTL="FALSE"           # installation flag
FLGXMLFL="FALSE"           # create XML tool definition file for SCRAM?
FGLKEEPT="FALSE"           # keep SHERPA source code tree?
FLGMCORE="FALSE"           # use multiple cores for compilation


# get & evaluate options
while getopts :v:d:m:l:p:F:W:P:Y:C:M:L:fSTADIXZKh OPT
do
  case $OPT in
  v) SHERPAVER=$OPTARG ;;
  d) IDIR=$OPTARG ;;
  f) FLAGS=TRUE &&
      OPTHEPMC=${OPTHEPMC}" -f" && OPTLHAPDF=${OPTLHAPDF}" -f" ;;
  F) FIXES=TRUE && FDIR=$OPTARG ;;
  m) HEPMC=TRUE && HVER=$OPTARG ;;
  M) OPTHEPMC=${OPTHEPMC}" "$OPTARG ;;
  l) LHAPDF=TRUE && LVER=$OPTARG ;;
  L) OPTLHAPDF=${OPTLHAPDF}" "$OPTARG ;;
  P) PDFSET=$PDFSET" "$OPTARG ;;
  S) LINKPDF=TRUE ;;
  p) PATCHES=TRUE && PDIR=$OPTARG ;;
  T) SHCFLAGS=${SHCFLAGS}" --enable-multithread" &&
      SHMFLAGS=${SHMFLAGS}" --copt --enable-multithread" ;;
  A) SHCFLAGS=${SHCFLAGS}" --enable-analysis" &&
      SHMFLAGS=${SHMFLAGS}" --copt --enable-analysis" ;;
  W) SHERPAWEBLOCATION=$OPTARG ;;
  Y) SHERPAFILE=$OPTARG ;;
  C) LVLCLEAN=$OPTARG ;;
  D) FLGDEBUG=TRUE &&
      OPTHEPMC=${OPTHEPMC}" -D" && OPTLHAPDF=${OPTLHAPDF}" -D" ;;
  I) FLGINSTL=TRUE ;;
  X) FLGXMLFL=TRUE &&
      OPTHEPMC=${OPTHEPMC}" -X" && OPTLHAPDF=${OPTLHAPDF}" -X" ;;
  Z) FLGMCORE=TRUE ;;
  K) FLGKEEPT=TRUE ;;
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
MSI=$SCRIPTPATH
shpatchfile="sherpa_patches_"${SHERPAVER}".tgz" # official patches for current SHERPA version
shfixfile="sherpa_fixes_"${SHERPAVER}".tgz"     # fixes for current SHERPA version
shshifile="install_sherpa.sh"        # this script
shhmifile="install_hepmc2.sh"        # script for HepMC2 installation
shlhifile="install_lhapdf.sh"        # script for LHAPDF installation


# analyze SHERPA version
va=`echo ${SHERPAVER} | cut -f1 -d"."`
vb=`echo ${SHERPAVER} | cut -f2 -d"."`
vc=`echo ${SHERPAVER} | cut -f3 -d"."`
#echo " <D> version breakdown: "$va" "$vb" "$vc

#http://www.hepforge.org/archive/sherpa/SHERPA-MC-1.2.0.tar.gz
# set SHERPA (HepMC2,LHAPDF) download location
if [ "$SHERPAWEBLOCATION" = "" ]; then
  SHERPAWEBLOCATION="http://www.hepforge.org/archive/sherpa"
  FLOC=" "
else
  if [ -e ${SHERPAWEBLOCATION} ]; then   # is the location a local subdirectory?
    if [ -d ${SHERPAWEBLOCATION} ]; then
      cd ${SHERPAWEBLOCATION}; SHERPAWEBLOCATION=`pwd`; cd ${HDIR}
##      SHERPAWEBLOCATION=$PWD"/"${SHERPAWEBLOCATION}
    fi
  fi
  FLOC=" -W "${SHERPAWEBLOCATION}
fi
if [ "$SHERPAFILE" = "" ]; then
  SHERPAFILE="Sherpa-"${SHERPAVER}".tar.gz"
  if [ $va -ge 1 ] && [ $vb -ge 2 ]; then
    SHERPAFILE="SHERPA-MC-"${SHERPAVER}".tar.gz"
  elif [ $va -ge 2 ]; then
    SHERPAFILE="SHERPA-MC-"${SHERPAVER}".tar.gz"
  fi
fi


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
echo "  -> SHERPA location: '"${SHERPAWEBLOCATION}"'"
echo "  -> SHERPA file name: '"${SHERPAFILE}"'"
echo "  -> cleaning level: '"${LVLCLEAN}"'"
echo "  -> debugging mode: '"${FLGDEBUG}"'"
echo "  -> CMSSW override: '"${FLGXMLFL}"'"
echo "  -> keep sources:   '"${FLGKEEPT}"'"
echo "  -> use makeinstall: '"${FLGINSTL}"'"
echo "  -> use multiple CPU cores: '"${FLGMCORE}"'"
echo "  -> HepMC2: '"${HEPMC}"', version '"${HVER}"'"
echo "  ->   options: "${OPTHEPMC}
echo "  -> LHAPDF: '"${LHAPDF}"', version '"${LVER}"'"
echo "  ->   options: "${OPTLHAPDF}
echo "  ->   PDFsets: "${PDFSET}
echo "  -> link PDFsets: '"${LINKPDF}"'"


# forward flags to HepMC2/LHAPDF
IFLG=" "
if [ ${LVLCLEAN} -gt 0 ]; then
  IFLG=${IFLG}" -C "${LVLCLEAN}
fi


# set path to local SHERPA installation
export SHERPADIR=${IDIR}"/SHERPA-MC-"${SHERPAVER}
export SHERPAIDIR=${IDIR}"/SHERPA_"${SHERPAVER}

if [ "$HEPMC" = "TRUE" ]; then
 CTESTH=`echo ${HVER} | cut -c1`
 if [ "$HVER" = "CMSSW" ]; then
  if [ ! "$CMSSW_BASE" = "" ]; then
    newdir=""
    cd $CMSSW_BASE &&
    newdir=`scramv1 tool info hepmc | grep BASE | cut -f 2 -d "="`
    if [ "${newdir}" = "" ]; then
      echo " <E> no 'hepmc' tool defined in CMSSW, are you sure that"
      echo " <E>  1. the command 'scramv1' is available ?"
      echo " <E>  2. the path to your CMSSW is correct ?"
      echo " <E>  3. there exists a HEPMC2 package in your CMSSW ?"
      exit 0
    else
      HEPMC2DIR=${newdir}
    fi
    cd ${HDIR}
  else
    echo " "
  fi
 elif [ "$CTESTH" = "." ] || [ "$CTESTH" = "/" ]; then
  cd ${HVER}
  HEPMC2DIR=`pwd`
  cd ${HDIR}
 fi
fi

if [ "$LHAPDF" = "TRUE" ]; then
 CTESTL=`echo ${LVER} | cut -c1`
 if [ "$LVER" = "CMSSW" ]; then
  if [ ! "$CMSSW_BASE" = "" ]; then
    newdir=""
    cd $CMSSW_BASE &&
    newdir=`scramv1 tool info lhapdf | grep BASE | cut -f 2 -d "="`
    if [ "${newdir}" = "" ]; then
      echo " <E> no 'lhapdf' tool defined in CMSSW, are you sure that"
      echo " <E>  1. the command 'scramv1' is available ?"
      echo " <E>  2. the path to your CMSSW is correct ?"
      echo " <E>  3. there exists a LHAPDF package in your CMSSW ?"
      exit 0
    else
      LHAPDFDIR=${newdir}
    fi
    cd ${HDIR}
  else
    echo " "
  fi
 elif [ "$CTESTL" = "." ] || [ "$CTESTL" = "/" ]; then
  cd ${LVER}
  LHAPDFDIR=`pwd`
  cd ${HDIR}
 fi
fi




# check HepMC2 installation (if required)
if [ "$HEPMC" = "TRUE" ]; then
  OPTHEPMC=${OPTHEPMC}" -v "${HVER}" -d "${IDIR}" "${IFLG}
  if [ "${FLGMCORE}" = "TRUE" ]; then
    OPTHEPMC=${OPTHEPMC}" -Z"
  fi
  if [ ! "$HEPMC2DIR" = "" ]; then
    echo " -> HepMC2 directory is: "${HEPMC2DIR}
    if [ ! -e ${HEPMC2DIR} ]; then
      echo " -> ... and does not exist: installing HepMC2..."
      echo " -> ... with command "${MSI}/${shhmifile} ${FLOC} ${OPTHEPMC}
      ${MSI}/${shhmifile} ${FLOC} ${OPTHEPMC}
    else
      echo " -> ... and exists: Installation cancelled!"
    fi
  else
    export HEPMC2IDIR=${IDIR}"/HEPMC_"${HVER}
    echo " -> no HepMC2 directory specified, trying installation"
    echo "     into "${HEPMC2IDIR}
    echo "     with command "${MSI}/${shhmifile} ${FLOC} ${OPTHEPMC}
    ${MSI}/${shhmifile} ${FLOC} ${OPTHEPMC}
    export HEPMC2DIR=${HEPMC2IDIR}
  fi
  SHCFLAGS=${SHCFLAGS}" --enable-hepmc2="${HEPMC2DIR}
  SHMFLAGS=${SHMFLAGS}" --copt --enable-hepmc2="${HEPMC2DIR}
fi



# check LHAPDF installation (if required)
PATCHLHAPDF=FALSE
FIXLHAPDF=FALSE
if [ "$LHAPDF" = "TRUE" ]; then
  OPTLHAPDF=${OPTLHAPDF}" -v "${LVER}" -d "${IDIR}" "${IFLG}
  if [ "${FLGMCORE}" = "TRUE" ]; then
    OPTLHAPDF=${OPTLHAPDF}" -Z"
  fi
  for pdfs in $PDFSET; do
    OPTLHAPDF=${OPTLHAPDF}" -P "${pdfs}
  done
  if [ ! "$LHAPDFDIR" = "" ]; then
    echo " -> LHAPDF directory is: "${LHAPDFDIR}
    if [ ! -e ${LHAPDFDIR} ]; then
      echo " -> ... and does not exist: installing LHAPDF..."
      echo " -> ... with command "${MSI}/${shlhifile} ${FLOC} ${OPTLHAPDF}
      ${MSI}/${shlhifile} ${FLOC} ${OPTLHAPDF}
    else
      echo " -> ... and exists: Installation cancelled!"
    fi
  else
    export LHAPDFIDIR=${IDIR}"/LHAPDF_"${LVER}
    echo " -> no LHAPDF directory specified, trying installation"
    echo "     into "${LHAPDFIDIR}
    echo "     with command "${MSI}/${shlhifile} ${FLOC} ${OPTLHAPDF}
    ${MSI}/${shlhifile} ${FLOC} ${OPTLHAPDF}
    export LHAPDFDIR=${LHAPDFIDIR}
  fi
  PATCHLHAPDF=TRUE
  FIXLHAPDF=TRUE
  SHCFLAGS=${SHCFLAGS}" --enable-lhapdf="${LHAPDFDIR}
  SHMFLAGS=${SHMFLAGS}" --copt --enable-lhapdf="${LHAPDFDIR}
fi


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
M_FCFLAGS=""
M_FFLAGS=""
M_CXXFLAGS=""
M_LDFLAGS=""
M_FFLGS2=""
M_CXXFLGS2=""
CF32BIT=""
if [ "$FLAGS" = "TRUE" ]; then
  CF32BIT="-m32"
  M_CFLAGS="CFLAGS="${CF32BIT}
  M_FCFLAGS="FCFLAGS="${CF32BIT}
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
#CONFFLG=${M_CFLAGS}" "${M_FFLAGS}" "${M_CXXFLAGS}" "${M_LDFLAGS}
CONFFLG=${M_CFLAGS}" "${M_FCFLAGS}" "${M_FFLAGS}" "${M_CXXFLAGS}" "${M_LDFLAGS}
if [ "$FLAGS" = "TRUE" ]; then
  M_CFLAGS=" --copt "${M_CFLAGS}
  M_LDFLAGS=" --copt "${M_LDFLAGS}
fi
#MAKEFLG=${M_CFLAGS}" "${M_LDFLAGS}" "${M_CXXFLGS2}" "${M_FFLGS2}
MAKEFLG=${M_CFLAGS}" "${M_FCFLAGS}" "${M_LDFLAGS}" "${M_CXXFLGS2}" "${M_FFLGS2}
SHCFLAGS=${SHCFLAGS}" "${CONFFLG}
SHMFLAGS=${SHMFLAGS}" "${MAKEFLG}

POPTS=""
if [ "$FLGMCORE" = "TRUE" ]; then
    nprc=`cat /proc/cpuinfo | grep  -c processor`
    let nprc=$nprc+1
    if [ $nprc -gt 2 ]; then
      echo " <I> multiple CPU cores detected: "$nprc"-1"
      POPTS=" -j"$nprc" "
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

if [ "$FLGINSTL" = "FALSE" ]; then
#  if [ "$FIXES" = "TRUE" ] && [ "${SHERPAVER}" = "1.1.2" ]; then
  if [ "$FIXES" = "TRUE" ]; then
# fix: 06.01.2009
    autoreconf -fi
#
#    aclocal
#    autoheader
#    automake
#    autoconf
  fi
  echo " -> configuring SHERPA with flags: --prefix="${SHERPAIDIR}" "${SHCFLAGS}
  echo "./configure --prefix="${SHERPAIDIR}" "${SHCFLAGS} > ../sherpa_configr.cmd
  ./configure --prefix=${SHERPAIDIR} ${SHCFLAGS} > ../sherpa_install.log 2>&1
  echo "-> making SHERPA with flags: "${CONFFLG}
  echo "make "${CONFFLG} >> ../sherpa_configr.cmd
  make ${POPTS} ${CONFFLG} >> ../sherpa_install.log 2>&1
  echo "-> making install SHERPA with flags: "${CONFFLG}
  echo "make install "${CONFFLG} >> ../sherpa_configr.cmd
  make install ${CONFFLG} >> ../sherpa_install.log 2>&1
  cd ${HDIR}
  if [ "$FLGKEEPT" = "TRUE" ]; then
    echo "-> keeping source code..."
  else
    rm -rf ${SHERPADIR}
  fi
else
  echo " -> installing SHERPA with flags: "${SHMFLAGS}
  echo "./TOOLS/makeinstall "${SHMFLAGS} > ../sherpa_install.cmd
  ./TOOLS/makeinstall ${POPTS} ${SHMFLAGS}
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
#    CNFFLG="CFLAGS=-m32 FFLAGS=-m32 CXXFLAGS=-m32 LDFLAGS=-m32"
#    MKEFLG="CFLAGS=-m32 FFLAGS=-m32 CXXFLAGS=\"-O2 -m32\" LDFLAGS=-m32"
    CNFFLG="CFLAGS=-m32 FCFLAGS=-m32 FFLAGS=-m32 CXXFLAGS=-m32 LDFLAGS=-m32"
    MKEFLG="CFLAGS=-m32 FCFLAGS=-m32 FFLAGS=-m32 CXXFLAGS=\"-O2 -m32\" LDFLAGS=-m32"
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
cd ${HDIR}


# create XML file for SCRAM
if [ "${FLGXMLFL}" = "TRUE" ]; then
#  xmlfile=sherpa.xml
  xmlfile="sherpa_"${SHERPAVER}".xml"
  echo " <I> creating Sherpa tool definition XML file"
  if [ -e ${xmlfile} ]; then rm ${xmlfile}; fi; touch ${xmlfile}
  echo "  <tool name=\"Sherpa\" version=\""${SHERPAVER}"\">" >> ${xmlfile}
  tmppath=`find ${SHERPADIR} -type f -name libSherpaMain.so\*`
  tmpcnt=`echo ${tmppath} | grep -o "/" | grep -c "/"`
  tmppath=`echo ${tmppath} | cut -f 0-${tmpcnt} -d "/"`
  for LIB in `cd ${tmppath}; ls *.so | cut -f 1 -d "." | sed -e 's/lib//'; cd ${HDIR}`; do
    echo "    <lib name=\""${LIB}"\"/>" >> ${xmlfile}
  done
  echo "    <client>" >> ${xmlfile}
  echo "      <Environment name=\"SHERPA_BASE\" value=\""${SHERPADIR}"\"/>" >> ${xmlfile}
  echo "      <Environment name=\"BINDIR\" default=\"\$SHERPA_BASE/bin\"/>" >> ${xmlfile}
  echo "      <Environment name=\"LIBDIR\" default=\"\$SHERPA_BASE/lib/SHERPA-MC\"/>" >> ${xmlfile}
  echo "      <Environment name=\"INCLUDE\" default=\"\$SHERPA_BASE/include/SHERPA-MC\"/>" >> ${xmlfile}
  echo "    </client>" >> ${xmlfile}
  echo "    <runtime name=\"LD_LIBRARY_PATH\" value=\"\$SHERPA_BASE/lib/SHERPA-MC\" type=\"path\"/>" >> ${xmlfile}
  echo "    <runtime name=\"CMSSW_FWLITE_INCLUDE_PATH\" value=\"\$SHERPA_BASE/include\" type=\"path\"/>" >> ${xmlfile}
  echo "    <runtime name=\"SHERPA_SHARE_PATH\" value=\"\$SHERPA_BASE/share/SHERPA-MC\" type=\"path\"/>" >> ${xmlfile}
  echo "    <runtime name=\"SHERPA_INCLUDE_PATH\" value=\"\$SHERPA_BASE/include/SHERPA-MC\" type=\"path\"/>" >> ${xmlfile}
  echo "    <use name=\"HepMC\"/>" >> ${xmlfile}
  echo "    <use name=\"lhapdf\"/>" >> ${xmlfile}
  echo "  </tool>" >> ${xmlfile}
  if [ ! "$PWD" = "${HDIR}" ]; then
    mv ${xmlfile} ${HDIR}/
  fi

  if [ ! "$CMSSW_BASE" = "" ]; then
    cd $CMSSW_BASE
    tmpsha=`scramv1 tool info sherpa | grep "SHERPA_BASE" | cut -f2 -d"="`
    tmpxml=`find $CMSSW_BASE/config/ -type f -name sherpa.xml -printf %h`
    echo " <I>"
    echo " <I> SHERPA version currently being used: "${tmpsha}
    echo " <I> ...defined in "${tmpxml}
    cd ${tmpxml}; tmpxml=$PWD; cd ${HDIR}
    echo " <I>"
    echo " <I> If you want to override this version with the freshly produced "${xmlfile}","
    echo " <I> ...please type (something like) the following commands:"
    echo " <I>"
    echo "       cd $CMSSW_BASE"
    echo "       scramv1 tool remove sherpa"
    echo "       cp ${HDIR}/${xmlfile} ${tmpxml}/"
    echo "       scramv1 setup sherpa"
    echo "       cd -"
    echo " <I>"
  fi

fi
cd ${HDIR}


### write these override commands into a script
if [ "${FLGXMLFL}" = "TRUE" ]; then
  if [ `echo $SHELL | grep -c -i csh` -gt 0 ]; then # (t)csh
    overrscr="Z_OVERRIDE.csh"
    shelltype="#!/bin/csh"
    setvarcmd="set "
    exportcmd="setenv "
    exporteqs=" "
  elif [ `echo $SHELL | grep -c -i bash` -gt 0 ]; then # bash
    overrscr="Z_OVERRIDE.sh"
    shelltype="#!/bin/bash"
    setvarcmd=""
    exportcmd="export "
    exporteqs="="
  else
    echo " <E> unknown shell type, stopping"
    exit 1
  fi

  if [ -e ${overrscr} ]; then
    rm ${overrscr}
  fi
  touch ${overrscr}
  echo ${shelltype} >> ${overrscr}
  echo "cd \$CMSSW_BASE" >> ${overrscr}
  lxmlfile="lhapdf_"${LVER}".xml"
  if [ "$LHAPDF" = "TRUE" ] && [ -e ${HDIR}/${lxmlfile} ]; then
    echo "scramv1 tool remove lhapdf" >> ${overrscr}
#    echo ${setvarcmd}"tmpxml=\`find \$CMSSW_BASE/config/ -type f -name lhapdf.xml -printf %h\`" >> ${overrscr}
#    echo "cp ${HDIR}/${lxmlfile} \${tmpxml}/" >> ${overrscr}
    echo ${setvarcmd}"tmpxml=\`find \$CMSSW_BASE/config/ -type f -name lhapdf.xml\`" >> ${overrscr}
    echo "cp ${HDIR}/${lxmlfile} \${tmpxml}" >> ${overrscr}
    echo "scramv1 setup lhapdf" >> ${overrscr}
#    echo "rm \${tmpxml}/lhapdf.xml" >> ${overrscr}
    echo ${setvarcmd}"newlibpath=\`find ${LHAPDFDIR}/ -name \\*.\*a -printf %h\\\\\n | head -1\`" >> ${overrscr}
    echo ${exportcmd}"LD_LIBRARY_PATH"${exporteqs}"\${newlibpath}:\$LD_LIBRARY_PATH" >> ${overrscr}
  fi
  hxmlfile="hepmc_"${HVER}".xml"
  if [ "$HEPMC" = "TRUE" ] && [ -e ${HDIR}/${hxmlfile} ]; then
    echo "scramv1 tool remove hepmc" >> ${overrscr}
#    echo ${setvarcmd}"tmpxml=\`find \$CMSSW_BASE/config/ -type f -name hepmc.xml -printf %h\`" >> ${overrscr}
#    echo "cp ${HDIR}/${hxmlfile} \${tmpxml}/" >> ${overrscr}
    echo ${setvarcmd}"tmpxml=\`find \$CMSSW_BASE/config/ -type f -name hepmc.xml\`" >> ${overrscr}
    echo "cp ${HDIR}/${hxmlfile} \${tmpxml}" >> ${overrscr}
    echo "scramv1 setup hepmc" >> ${overrscr}
#    echo "rm \${tmpxml}/hepmc.xml" >> ${overrscr}
    echo ${setvarcmd}"newlibpath=\`find ${HEPMC2DIR}/ -name \\*.\*a -printf %h\\\\\n | head -1\`" >> ${overrscr}
    echo ${exportcmd}"LD_LIBRARY_PATH"${exporteqs}"\${newlibpath}:\$LD_LIBRARY_PATH" >> ${overrscr}
  fi
  sxmlfile=${xmlfile}
  echo "scramv1 tool remove sherpa" >> ${overrscr}
#  echo ${setvarcmd}"tmpxml=\`find \$CMSSW_BASE/config/ -type f -name sherpa.xml -printf %h\`" >> ${overrscr}
#  echo "cp ${HDIR}/${sxmlfile} \${tmpxml}/" >> ${overrscr}
  echo ${setvarcmd}"tmpxml=\`find \$CMSSW_BASE/config/ -type f -name sherpa.xml\`" >> ${overrscr}
  echo "cp ${HDIR}/${sxmlfile} \${tmpxml}" >> ${overrscr}
  echo "scramv1 setup sherpa" >> ${overrscr}
#  echo "rm \${tmpxml}/sherpa.xml" >> ${overrscr}
  echo ${setvarcmd}"newlibpath=\`find ${SHERPADIR}/ -name \\*.so -printf %h\\\\\n | head -1\`" >> ${overrscr}
  echo ${exportcmd}"LD_LIBRARY_PATH"${exporteqs}"\${newlibpath}:\$LD_LIBRARY_PATH" >> ${overrscr}
  echo "cd -" >> ${overrscr}
  chmod u+x ${overrscr}

  echo " <I> ===> you can find these override commands collected in the script "${overrscr}
  echo " <I> ===> just type 'source ./"${overrscr}"'"
fi


# summarize installation
echo " <I> Summary of the SHERPA installation:"
if [ "$HEPMC" = "TRUE" ]; then
echo " <I> HepMC2 version "${HEPMC2VER}" installed in "${HEPMC2DIR}
fi
if [ "$LHAPDF" = "TRUE" ]; then
echo ""
echo ""
echo ""
echo " <I> LHAPDF version "${LHAPDFVER}" installed in "${LHAPDFDIR}
echo " <I>  -> before using SHERPA please define"
echo " <I>  -> export LHAPATH="${pdfdir}
echo ""
echo ""
echo ""
fi
echo " <I> SHERPA version "${SHERPAVER}" installed in "${SHERPADIR}


