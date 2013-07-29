
#!/bin/bash
#
#  file:        install_sherpa.sh
#  description: BASH script for the installation of the SHERPA MC generator,
#               downloads and installs SHERPA, HepMC2, LHAPDF, BlackHat and qd (if required),
#               can be used standalone or called from other scripts,
#  uses:        install_hepmc2.sh
#               install_lhapdf.sh
#               install_blackhat.sh
#               install_qd.sh
#               SHERPA patches/fixes [see below]
#
#  author:      Markus Merschmeyer, RWTH Aachen University
#  date:        2013/05/23
#  version:     4.0
#

print_help() {
    echo "" && \
    echo "install_sherpa version 4.0" && echo && \
    echo "options: -v  version    define SHERPA version ( "${SHERPAVER}" )" && \
    echo "         -d  path       define (SHERPA) installation directory" && \
    echo "                         -> ( "${IDIR}" )" && \
    echo "         -F             apply inofficial fixes (LHAPDF,...)  ( "${FIXES}" )" && \
    echo "                         from this path ( "${FDIR}" )" && \
    echo "         -p  path       apply official SHERPA patches ( "${PATCHES}" )" && \
    echo "                         from this path ( "${PDIR}" )" && \
    echo "         -m  version    request HepMC2 installation ( "${HEPMC}", "${HVER}" )" && \
    echo "         -M  options    special HepMC2 options ( "${OPTHEPMC}" )" && \
    echo "         -l  version    request LHAPDF installation ( "${LHAPDF}", "${LVER}" )" && \
    echo "         -L  options    special LHAPDF options ( "${OPTLHAPDF}" )" && \
    echo "         -P  name       automatically retrieve LHAPDF set ( "${PDFSET}" )" && \
#    echo "         -S             link (softlink) LHAPDF sets ( "${LINKPDF}" )" && \
#    echo "                         or do a hardcopy if not set" && \
    echo "         -q  version    request qd installation ( "${QD}", "${QDVER}" )" && \
    echo "         -Q  options    special qd options ( "${OPTQD}" )" && \
    echo "         -b  version    request BlackHat installation ( "${BH}", "${BHVER}" )" && \
    echo "         -B  options    special BlackHat options ( "${OPTBH}" )" && \
    echo "         -W  location   (web)location of SHERPA tarball ( "${SHERPAWEBLOCATION}" )" && \
    echo "         -Y  filename   file name of SHERPA tarball ( "${SHERPAFILE}" )" && \
    echo "         -C  level      cleaning level of SHERPA installation ("${LVLCLEAN}" )" && \
    echo "                         -> 0: nothing, 1: +objects, 2: +sourcecode" && \
    echo "         -c  option     add option for configure script ("${OPTCONFIG}" )" && \
    echo "         -T             enable multithreading"  && \
    echo "         -t             enable MPI" && \
    echo "         -A             enable analysis" && \
    echo "         -D             debug flag, compile with '-g' option ("${FLGDEBUG}" )" && \
    echo "         -X             create XML file for tool override in CMSSW ( "${FLGXMLFL}" )" && \
    echo "         -Z             use multiple CPU cores if available ( "${FLGMCORE}" )" && \
    echo "         -K             keep SHERPA source code tree after installation ( "${FLGKEEPT}" )" && \
    echo "         -h             display this help and exit" && echo
}

replace_tool() {
# $1 : tool name (e.g. 'qd')
# $2 : path to replacement XML file
# $3 : name of replacement XML file
# $4 : tool installation directory
# $5 : output file name
    echo "#################################################################" >> $5
    echo "" >> $5
    echo "tmptool=$1" >> $5
    echo "tmpfil=\${tmptool}.xml" >> $5
    echo "tmpxml=\`find \$CMSSW_BASE/config/ -type f -name \${tmpfil}\`" >> $5
    echo "if [ \"\${tmpxml}\" = \"\" ]; then" >> $5
    echo "  tmpxml=\`find \$CMSSW_BASE/config/ -type d -name available\`" >> $5
    echo "  tmpxml=\${tmpxml}/\${tmpfil}" >> $5
    echo "else" >> $5
    echo "  scramv1 tool remove \${tmptool}" >> $5
    echo "fi" >> $5
    echo "cp $2/$3 \${tmpxml}" >> $5
    echo "scramv1 setup \${tmptool}" >> $5
    echo "newlibpath=\`find $4 -name \\*.\*a -printf %h\\\\\n | head -1\`" >> $5
    echo "export LD_LIBRARY_PATH=\${newlibpath}:\$LD_LIBRARY_PATH" >> $5
    echo "" >> $5
    echo "#################################################################" >> $5
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
SHCFLAGS=""                # SHERPA compiler flags
HEPMC="FALSE"              # install HepMC2
HVER="2.05.00"             # HepMC2 version to be installed
OPTHEPMC=""                # special HepMC2 installation options
LHAPDF="FALSE"             # install LHAPDF
OPTLHAPDF=""               # special LHAPDF installation options
PDFSET=""                  # name of PDFset to download
#LINKPDF="FALSE"            # link (softlink) LHAPDF sets
LVER="5.6.0"               # LHAPDF version to be installed
QD="FALSE"                 # install qd
QDVER="2.3.13"             # qd version to be installed
OPTQD=""                   # special qd installation options
BH="FALSE"                 # install BlackHat
BHVER="0.9.9"              # BlackHat version to be installed
OPTBH=""                   # special BlackHat installation options
FIXES="FALSE"              # apply inofficial fixes
FDIR="./"                  # path containing fixes
PATCHES="FALSE"            # apply SHERPA patches
PDIR="./"                  # path containing patches
SHERPAWEBLOCATION=""       # (web)location of SHERPA tarball
SHERPAFILE=""              # file name of SHERPA tarball
LVLCLEAN=0                 # cleaning level (0-2)
OPTCONFIG=""               # option(s) for configure script
FLGDEBUG="FALSE"           # debug flag for compilation
FLGXMLFL="FALSE"           # create XML tool definition file for SCRAM?
FLGKEEPT="FALSE"           # keep the source code tree?
FLGMCORE="FALSE"           # use multiple cores for compilation


# get & evaluate options
while getopts :v:d:m:l:q:b:p:F:W:P:Y:C:c:M:L:Q:B:STtADXZKh OPT
do
  case $OPT in
  v) SHERPAVER=$OPTARG ;;
  d) IDIR=$OPTARG ;;
  F) FIXES=TRUE && FDIR=$OPTARG ;;
  m) HEPMC=TRUE && HVER=$OPTARG ;;
  M) OPTHEPMC=${OPTHEPMC}" "$OPTARG ;;
  l) LHAPDF=TRUE && LVER=$OPTARG ;;
  L) OPTLHAPDF=${OPTLHAPDF}" "$OPTARG ;;
  P) PDFSET=$PDFSET" "$OPTARG ;;
#  S) LINKPDF=TRUE ;;
  q) QD=TRUE && QDVER=$OPTARG ;;
  Q) OPTQD=${OPTQD}" "$OPTARG ;;
  b) BH=TRUE && BHVER=$OPTARG ;;
  B) OPTBH=${OPTBH}" "$OPTARG ;;
  p) PATCHES=TRUE && PDIR=$OPTARG ;;
  T) SHCFLAGS=${SHCFLAGS}" --enable-multithread" ;;
  t) SHCFLAGS=${SHCFLAGS}" --enable-mpi" ;;
  A) SHCFLAGS=${SHCFLAGS}" --enable-analysis" ;;
  W) SHERPAWEBLOCATION=$OPTARG ;;
  Y) SHERPAFILE=$OPTARG ;;
  C) LVLCLEAN=$OPTARG ;;
  c) SHCFLAGS=${SHCFLAGS}" "$OPTARG ;;
  D) FLGDEBUG=TRUE ;;
  X) FLGXMLFL=TRUE ;;
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
shqdifile="install_qd.sh"            # script for qd installation
shbhifile="install_blackhat.sh"      # script for BlackHat installation


# set SHERPA (HepMC2,LHAPDF) download location
if [ "$SHERPAWEBLOCATION" = "" ]; then
  SHERPAWEBLOCATION="http://www.hepforge.org/archive/sherpa"
  FLOC=" "
else
  if [ -e ${SHERPAWEBLOCATION} ]; then   # is the location a local subdirectory?
    if [ -d ${SHERPAWEBLOCATION} ]; then
      cd ${SHERPAWEBLOCATION}; SHERPAWEBLOCATION=`pwd`; cd ${HDIR}
    fi
  fi
  FLOC=" -W "${SHERPAWEBLOCATION}
fi
if [ "$SHERPAFILE" = "" ]; then
  SHERPAFILE="SHERPA-MC-"${SHERPAVER}".tar.gz"
fi


# make SHERPA version a global variable
export SHERPAVER=${SHERPAVER}
export HEPMC2VER=${HVER}
export LHAPDFVER=${LVER}
export BHVER=${BHVER}
export QDVER=${QDVER}

# always use absolute path names...
cd ${IDIR}; IDIR=`pwd`; cd ${HDIR}
cd ${PDIR}; PDIR=`pwd`; cd ${HDIR}
cd ${FDIR}; FDIR=`pwd`; cd ${HDIR}

# print basic setup information
echo " SHERPA (HepMC2,LHAPDF) installation: "
echo "  -> SHERPA version: '"${SHERPAVER}"'"
echo "  -> installation directory: '"${IDIR}"'"
echo "  -> SHERPA patches: '"${PATCHES}"' in '"${PDIR}"'"
echo "  -> SHERPA fixes: '"${FIXES}"' in '"${FDIR}"'"
echo "  -> SHERPA location: '"${SHERPAWEBLOCATION}"'"
echo "  -> SHERPA file name: '"${SHERPAFILE}"'"
echo "  -> cleaning level: '"${LVLCLEAN}"'"
echo "  -> configure options: '"${SHCFLAGS}"'"
echo "  -> debugging mode: '"${FLGDEBUG}"'"
echo "  -> CMSSW override: '"${FLGXMLFL}"'"
echo "  -> keep sources:   '"${FLGKEEPT}"'"
echo "  -> use multiple CPU cores: '"${FLGMCORE}"'"
echo "  -> HepMC2: '"${HEPMC}"', version '"${HVER}"'"
echo "  ->   options: "${OPTHEPMC}
echo "  -> LHAPDF: '"${LHAPDF}"', version '"${LVER}"'"
echo "  ->   options: "${OPTLHAPDF}
echo "  ->   PDFsets: "${PDFSET}
#echo "  -> link PDFsets: '"${LINKPDF}"'"
echo "  -> qd: '"${QD}"', version '"${QDVER}"'"
echo "  ->   options: "${OPTQD}
echo "  -> BlackHat: '"${BH}"', version '"${BHVER}"'"
echo "  ->   options: "${OPTBH}



# set basic CXX, CXXFLAGS and LDFLAGS & other flags
tmpcxx="g++"
#tmpcxx="g++44"
tmpcxxflg="-O2 -m64 -std=c++0x -fuse-cxa-atexit"
#tmpcxxflg="-O2 -m64"
if [ "$FLGDEBUG" = "TRUE" ]; then
  tmpcxxflg=${tmpcxxflg}" -g"
fi
tmpldflg="-ldl"
tmpconfg="--disable-silent-rules"
if [ "${tmpcxx}" != "" ]; then
  if [ "${CXX}" != "" ]; then
    export CXX=${tmpcxx}" "${CXX}
  else
    export CXX=${tmpcxx}
  fi
  echo " <I> basic CXX option(s): "${tmpcxx}
fi
if [ "${tmpcxxflg}" != "" ]; then
  if [ "${CXXFLAGS}" != "" ]; then
    export CXXFLAGS=${tmpcxxflg}" "${CXXFLAGS}
  else
    export CXXFLAGS=${tmpcxxflg}
  fi
  echo " <I> basic CXXFLAGS: "${tmpcxxflg}
fi
if [ "${tmpldflg}" != "" ]; then
  if [ "${LDFLAGS}" != "" ]; then
    export LDFLAGS=${tmpldflg}" "${LDFLAGS}
  else
    export LDFLAGS=${tmpldflg}
  fi
  echo " <I> basic LDLFAGS: "${tmpldflg}
fi
echo " <I> basic ./configure option(s): "${tmpconfg}
SHCFLAGS=${SHCFLAGS}" "${tmpconfg}
# initialize custom flags
M_CXX=""
M_CXXFLAGS=""
M_LDFLAGS=""



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
    echo " <E> probably 'cmsenv' was not executed"
    exit 0
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
    echo " <E> probably 'cmsenv' was not executed"
    exit 0
  fi
 elif [ "$CTESTL" = "." ] || [ "$CTESTL" = "/" ]; then
  cd ${LVER}
  LHAPDFDIR=`pwd`
  cd ${HDIR}
 fi
fi

if [ "$QD" = "TRUE" ]; then
 CTESTL=`echo ${QDVER} | cut -c1`
 if [ "$BHVER" = "CMSSW" ]; then QDVER="CMSSW"; fi
 if [ "$QDVER" = "CMSSW" ]; then
  if [ ! "$CMSSW_BASE" = "" ]; then
    newdir=""
    cd $CMSSW_BASE &&
    newdir=`scramv1 tool info qd | grep BASE | cut -f 2 -d "="`
    if [ "${newdir}" = "" ]; then
      echo " <E> no 'qd' tool defined in CMSSW, are you sure that"
      echo " <E>  1. the command 'scramv1' is available ?"
      echo " <E>  2. the path to your CMSSW is correct ?"
      echo " <E>  3. there exists a qd package in your CMSSW ?"
      exit 0
    else
      QDDIR=${newdir}
    fi
    cd ${HDIR}
  else
    echo " <E> probably 'cmsenv' was not executed"
    exit 0
  fi
 elif [ "$CTESTL" = "." ] || [ "$CTESTL" = "/" ]; then
  cd ${QDVER}
  QDDIR=`pwd`
  cd ${HDIR}
 fi
fi

if [ "$BH" = "TRUE" ]; then
 CTESTL=`echo ${BHVER} | cut -c1`
 if [ "$BHVER" = "CMSSW" ]; then
  if [ ! "$CMSSW_BASE" = "" ]; then
    newdir=""
    cd $CMSSW_BASE &&
    newdir=`scramv1 tool info blackhat | grep BASE | cut -f 2 -d "="`
    if [ "${newdir}" = "" ]; then
      echo " <E> no 'blackhat' tool defined in CMSSW, are you sure that"
      echo " <E>  1. the command 'scramv1' is available ?"
      echo " <E>  2. the path to your CMSSW is correct ?"
      echo " <E>  3. there exists a BlackHat package in your CMSSW ?"
      exit 0
    else
      BHDIR=${newdir}
    fi
    cd ${HDIR}
  else
    echo " <E> probably 'cmsenv' was not executed"
    exit 0
  fi
 elif [ "$CTESTL" = "." ] || [ "$CTESTL" = "/" ]; then
  cd ${BHVER}
  BHDIR=`pwd`
  cd ${HDIR}
 fi
fi



# check HepMC2 installation (if required)
if [ "$HEPMC" = "TRUE" ]; then
  OPTHEPMC=${OPTHEPMC}" -v "${HVER}" -d "${IDIR}
  if [ "${FLGMCORE}" = "TRUE" ]; then OPTHEPMC=${OPTHEPMC}" -Z"; fi
  if [ "${FLGXMLFL}" = "TRUE" ]; then OPTHEPMC=${OPTHEPMC}" -X"; fi
  if [ "${FLGDEBUG}" = "TRUE" ]; then OPTHEPMC=${OPTHEPMC}" -D"; fi
  if [ "${FLGKEEPT}" = "TRUE" ]; then OPTHEPMC=${OPTHEPMC}" -K"; fi
  if [ ${LVLCLEAN} -gt 0 ]; then OPTHEPMC=${OPTHEPMC}" -C "${LVLCLEAN}; fi
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
fi


# check LHAPDF installation (if required)
PATCHLHAPDF=FALSE
FIXLHAPDF=FALSE
if [ "$LHAPDF" = "TRUE" ]; then
  OPTLHAPDF=${OPTLHAPDF}" -v "${LVER}" -d "${IDIR}
  if [ "${FLGMCORE}" = "TRUE" ]; then OPTLHAPDF=${OPTLHAPDF}" -Z"; fi
  if [ "${FLGXMLFL}" = "TRUE" ]; then OPTLHAPDF=${OPTLHAPDF}" -X"; fi
  if [ "${FLGDEBUG}" = "TRUE" ]; then OPTLHAPDF=${OPTLHAPDF}" -D"; fi
  if [ "${FLGKEEPT}" = "TRUE" ]; then OPTLHAPDF=${OPTLHAPDF}" -K"; fi
  if [ ${LVLCLEAN} -gt 0 ]; then OPTLHAPDF=${OPTLHAPDF}" -C "${LVLCLEAN}; fi
  for pdfs in $PDFSET; do OPTLHAPDF=${OPTLHAPDF}" -P "${pdfs}; done
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
fi


# check qd installation (if required)
if [ "$QD" = "TRUE" ]; then
  OPTQD=${OPTQD}" -v "${QDVER}" -d "${IDIR}
  if [ "${FLGMCORE}" = "TRUE" ]; then OPTQD=${OPTQD}" -Z"; fi
  if [ "${FLGXMLFL}" = "TRUE" ]; then OPTQD=${OPTQD}" -X"; fi
  if [ "${FLGDEBUG}" = "TRUE" ]; then OPTQD=${OPTQD}" -D"; fi
  if [ "${FLGKEEPT}" = "TRUE" ]; then OPTQD=${OPTQD}" -K"; fi
  if [ ${LVLCLEAN} -gt 0 ]; then OPTQD=${OPTQD}" -C "${LVLCLEAN}; fi
  if [ ! "$QDDIR" = "" ]; then
    echo " -> qd directory is: "${QDDIR}
    if [ ! -e ${QDDIR} ]; then
      echo " -> ... and does not exist: installing qd..."
      echo " -> ... with command "${MSI}/${shqdifile} ${FLOC} ${OPTQD}
      ${MSI}/${shqdifile} ${FLOC} ${OPTQD}
    else
      echo " -> ... and exists: Installation cancelled!"
    fi
  else
    export QDIDIR=${IDIR}"/QD_"${QDVER}
    echo " -> no qd directory specified, trying installation"
    echo "     into "${QDIDIR}
    echo "     with command "${MSI}/${shqdifile} ${FLOC} ${OPTQD}
    ${MSI}/${shqdifile} ${FLOC} ${OPTQD}
    export QDDIR=${QDIDIR}
  fi
  SHCFLAGS=${SHCFLAGS}
# update compiler & linker flags
  M_CXXFLAGS=${M_CXXFLAGS}" -I"${QDDIR}"/include"
  M_LDFLAGS=${M_LDFLAGS}" -L"${QDDIR}"/lib"
fi


# check BlackHat installation (if required)
if [ "$BH" = "TRUE" ]; then
if [ ! "$QDDIR" = "" ] && [ -e ${QDDIR} ]; then
  OPTBH=${OPTBH}" -v "${BHVER}" -d "${IDIR}" -Q "${QDDIR}
  if [ "${FLGMCORE}" = "TRUE" ]; then OPTBH=${OPTBH}" -Z"; fi
  if [ "${FLGXMLFL}" = "TRUE" ]; then OPTBH=${OPTBH}" -X"; fi
  if [ "${FLGDEBUG}" = "TRUE" ]; then OPTBH=${OPTBH}" -D"; fi
  if [ "${FLGKEEPT}" = "TRUE" ]; then OPTBH=${OPTBH}" -K"; fi
  if [ ${LVLCLEAN} -gt 0 ]; then OPTBH=${OPTBH}" -C "${LVLCLEAN}; fi
  if [ ! "$BHDIR" = "" ]; then
    echo " -> BlackHat directory is: "${BHDIR}
    if [ ! -e ${BHDIR} ]; then
      echo " -> ... and does not exist: installing BlackHat..."
      echo " -> ... with command "${MSI}/${shbhifile} ${FLOC} ${OPTBH}
      ${MSI}/${shbhifile} ${FLOC} ${OPTBH}
    else
      echo " -> ... and exists: Installation cancelled!"
    fi
  else
    export BHIDIR=${IDIR}"/BH_"${BHVER}
    echo " -> no BlackHat directory specified, trying installation"
    echo "     into "${BHIDIR}
    echo "     with command "${MSI}/${shbhifile} ${FLOC} ${OPTBH}
    ${MSI}/${shbhifile} ${FLOC} ${OPTBH}
    export BHDIR=${BHIDIR}
  fi
  SHCFLAGS=${SHCFLAGS}" --enable-blackhat="${BHDIR}
# update compiler & linker flags
  CXXFLAGS=${CXXFLAGS}" -I"${BHDIR}"/include"
  M_LDFLAGS=${M_LDFLAGS}" -L"${BHDIR}"/lib/blackhat"
else
  echo " <E> qd directory is empty, cannot install BlackHat"
  exit 0
fi
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
MOPTS=""
POPTS=""
if [ "$FLGMCORE" = "TRUE" ]; then
    nprc=`cat /proc/cpuinfo | grep  -c processor`
    if [ $nprc -gt 1 ]; then
      echo " <I> multiple CPU cores detected: "$nprc
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
export SHERPA_INCLUDE_PATH=${SHERPAIDIR}"/include/SHERPA-MC"
export SHERPA_SHARE_PATH=${SHERPAIDIR}"/share/SHERPA-MC"
if [ -e ${SHERPADIR}/bin/Sherpa ]; then
  echo " <E> installed SHERPA exists, clean up!"
  exit 0
fi

  if [ "$FIXES" = "TRUE" ] || [ "$PATCHES" = "TRUE" ]; then
    echo " <I> doing autoreconf"
    autoreconf -i --force -v
  fi
#
  if [ "${M_CXX}" != "" ]; then
    if [ "${CXX}" != "" ]; then
      export CXX=${M_CXX}" "${CXX}
    else
      export CXX=${M_CXX}
    fi
    echo " <I> CXX variable content: "${CXX}
  fi
  if [ "${M_CXXFLAGS}" != "" ]; then
    if [ "${CXXFLAGS}" != "" ]; then
      export CXXFLAGS=${M_CXXFLAGS}" "${CXXFLAGS}
    else
      export CXXFLAGS=${M_CXXFLAGS}
    fi
    echo " <I> CXXFLAGS variable content: "${CXXFLAGS}
  fi
  if [ "${M_LDFLAGS}" != "" ]; then
    if [ "${LDFLAGS}" != "" ]; then
      export LDFLAGS=${M_LDFLAGS}" "${LDFLAGS}
    else
      export LDFLAGS=${M_LDFLAGS}
    fi
    echo " <I> LDFLAGS variable content: "${LDFLAGS}
  fi
  echo " <I> configuring with prefix: "${SHERPAIDIR}
  echo " <I> ... and options: "${SHCFLAGS}
#  ./configure --prefix=${SHERPAIDIR} ${SHCFLAGS} > ../sherpa_install.log 2>&1
  ./configure --prefix=${SHERPAIDIR} ${SHCFLAGS}  && \
#
  echo " <I> make Sherpa with options: "${POPTS} && \
#  make ${POPTS} >> ../sherpa_install.log 2>&1
  make ${POPTS} && \
#
  echo " <I> make install Sherpa" && \
#  make install >> ../sherpa_install.log 2>&1
  make install
  cd ${HDIR}
  if [ "$FLGKEEPT" = "TRUE" ]; then
    echo " <I> keeping source code..."
  else
    rm -rf ${SHERPADIR}
  fi
export SHERPADIR=${SHERPAIDIR}
cd ${HDIR}


##FIXME???
# get LHAPDFs into SHERPA... (now with symbolic links)
#if [ "$LHAPDF" = "TRUE" ]; then
#  pdfdir=`find ${LHAPDFDIR} -type d -name PDFsets`
#  if [ ! -e ${pdfdir} ]; then
#    echo " <E> PDFsets of LHAPDF not found, stopping..."
#    exit 1
#  fi
#  if [ "${LINKPDF}" = "TRUE" ] && [ ! "${pdfdir}" = "" ]; then
#    ln -s ${pdfdir} ${SHERPADIR}/share/SHERPA-MC/PDFsets
#  else
#    cp -r ${pdfdir} ${SHERPADIR}/share/SHERPA-MC/
#  fi
#fi
#cd ${HDIR}


# create XML file for SCRAM
if [ "${FLGXMLFL}" = "TRUE" ]; then
#  xmlfile=sherpa.xml
  xmlfile="sherpa_"${SHERPAVER}".xml"
  echo " <I> creating Sherpa tool definition XML file"
  if [ -e ${xmlfile} ]; then rm ${xmlfile}; fi; touch ${xmlfile}
  echo "  <tool name=\"Sherpa\" version=\""${SHERPAVER}"\">" >> ${xmlfile}
  tmppath=`find ${SHERPADIR} -type f -name libSherpaMain.so\*`
  tmpcnt=`echo ${tmppath} | grep -o "/" | grep -c "/"`
  tmppath=`echo ${tmppath} | cut -f 1-${tmpcnt} -d "/"`
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
  overrscr="Z_OVERRIDE.sh"
  shelltype="#!/bin/bash"
  if [ -e ${overrscr} ]; then
    rm ${overrscr}
  fi
  touch ${overrscr}
  echo ${shelltype} >> ${overrscr}
  echo "cd \$CMSSW_BASE" >> ${overrscr}
  lxmlfile="lhapdf_"${LVER}".xml"
  if [ "$LHAPDF" = "TRUE" ] && [ -e ${HDIR}/${lxmlfile} ]; then
    replace_tool lhapdf ${HDIR} ${lxmlfile} ${LHAPDFDIR} ${overrscr}
  fi
  hxmlfile="hepmc_"${HVER}".xml"
  if [ "$HEPMC" = "TRUE" ] && [ -e ${HDIR}/${hxmlfile} ]; then
    replace_tool hepmc ${HDIR} ${hxmlfile} ${HEPMC2DIR} ${overrscr}
  fi
  qxmlfile="qd_"${QDVER}".xml"
  if [ "$QD" = "TRUE" ] && [ -e ${HDIR}/${qxmlfile} ]; then
    replace_tool qd ${HDIR} ${qxmlfile} ${QDDIR} ${overrscr}
  fi
  bxmlfile="blackhat_"${BHVER}".xml"
  if [ "$BH" = "TRUE" ] && [ -e ${HDIR}/${bxmlfile} ]; then
    replace_tool blackhat ${HDIR} ${bxmlfile} ${BHDIR} ${overrscr}
  fi
  sxmlfile=${xmlfile}
  echo "scramv1 tool remove sherpa" >> ${overrscr}
  echo ${setvarcmd}"tmpfil=sherpa.xml" >> ${overrscr}
  echo ${setvarcmd}"tmpxml=\`find \$CMSSW_BASE/config/ -type f -name \${tmpfil}\`" >> ${overrscr}
  echo "if [ \"\${tmpxml}\" = \"\" ]; then" >> ${overrscr}
  echo "  ${setvarcmd}tmpxml=\`find \$CMSSW_BASE/config/ -type d -name selected\`" >> ${overrscr}
  echo "  ${setvarcmd}tmpxml=\${tmpxml}/\${tmpfil}" >> ${overrscr}
  echo "fi" >> ${overrscr}
  echo "cp ${HDIR}/${sxmlfile} \${tmpxml}" >> ${overrscr}
  echo "scramv1 setup sherpa" >> ${overrscr}
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
  echo ""
  echo " <I> HepMC2 version "${HEPMC2VER}" installed in "${HEPMC2DIR}
  echo ""
fi
if [ "$LHAPDF" = "TRUE" ]; then
  echo ""
  echo " <I> LHAPDF version "${LHAPDFVER}" installed in "${LHAPDFDIR}
  echo " <I>  -> before using SHERPA please define"
  echo " <I>  -> export LHAPATH="${pdfdir}
  echo ""
fi
if [ "$QD" = "TRUE" ]; then
  echo ""
  echo " <I> qd version "${QDVER}" installed in "${QDDIR}
  echo ""
fi
if [ "$BH" = "TRUE" ]; then
  echo ""
  echo " <I> BlackHat version "${BHVER}" installed in "${BHDIR}
  echo ""
fi
echo " <I> SHERPA version "${SHERPAVER}" installed in "${SHERPADIR}


