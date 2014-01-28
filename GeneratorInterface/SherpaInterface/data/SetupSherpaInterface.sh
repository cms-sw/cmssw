#!/bin/bash
#
#  file:        SetupSherpaInterface.sh
#  description: BASH script for the LOCAL installation of the SHERPA MC generator,
#               HepMC2 & LHAPDF (optional) and the CMSSW SHERPA interface
#  uses:        install_sherpa.sh
#               install_hepmc2.sh
#               install_lhapdf.sh
#               SHERPA patches (optional)
#               SHERPA fixes (optional)
#
#  author:      Markus Merschmeyer, RWTH Aachen
#  date:        2010/01/15
#  version:     2.5
#



# +-----------------------------------------------------------------------------------------------+
# function definitions
# +-----------------------------------------------------------------------------------------------+

function print_help() {
    echo "" && \
    echo "SetupSherpaInterface version 2.4" && echo && \
    echo "options: -i  path       installation directory for SHERPA,..." && \
    echo "                         -> ( "${instdir}" )" && \
    echo "         -p  path       location of required SHERPA patches/fixes" && \
    echo "                         -> ( "${patdir}" )" && \
    echo "         -d  path       location of the CMSSW directory" && \
    echo "                         -> ( "${cmsswd}" )" && \
    echo "         -o  option(s)  expert options ( "${xopt}" )" && \
    echo "         -m  mode       running mode ['LOCAL','GRID'] ( "${imode}" )" && \
    echo "         -l  path       location of SHERPA interface tarball" && \
    echo "                         -> ( "${ship}" )" && \
    echo "         -f  filename   name of SHERPA interface tarball ( "${shif}" )" && \
    echo "         -W  location   (optional) location of SHERPA tarball ( "${SHERPAWEBLOCATION}" )" && \
    echo "         -S  filename   (optional) file name of SHERPA tarball ( "${SHERPAFILE}" )" && \
    echo "         -C  level      cleaning level of (SHERPA) installation ("${LVLCLEAN}" )" && \
    echo "                         -> 0: nothing, 1: +objects, 2: +sourcecode" && \
    echo "         -D             debug flag, compile with '-g' option ("${FLGDEBUG}" )" && \
    echo "         -h             display this help and exit" && echo
}

# function to copy files from different locations (WWW, SE, local)
function file_get() {
# $1 : target path
# $2 : file name
# $3 : destination path
#
#  srmopt=" -debug -streams_num=1 "
  srmopt=" -streams_num=1 "
#
  tpath="./"
  fname="xxx.yyy"
  dpath="./"
  if [ $# -ge 1 ]; then
    tpath=$1
    if [ $# -ge 2 ]; then
      fname=$2
      if [ $# -ge 3 ]; then
        dpath=$3
      fi
    fi
  fi
  ILOC0=`echo ${tpath} | cut -f1 -d "/" | grep -c -i http`
  ILOC1=`echo ${tpath} | cut -f1 -d "/" | grep -c -i srm`
  cd ${dpath}
  if [ ${ILOC0} -gt 0 ]; then    # file is in WWW
    echo " <I> retrieving WWW file: "${fname}
    echo " <I>  from path: "${tpath}
    wget ${tpath}/${fname}
  elif [ ${ILOC1} -gt 0 ]; then  # file is in SE
    echo " <I> retrieving SE file: "${fname}
    echo " <I>  from path: "${tpath}
#    srmcp ${tpath}/${fname} file:///${PWD}/${fname}
    srmcp ${srmopt} ${tpath}/${fname} file:///${PWD}/${fname}
  else                           # local file
    echo " <I> copying local file: "${fname}
    echo " <I>  from path: "${tpath}
    cp ${tpath}/${fname} ./
  fi
  cd -
}



# +-----------------------------------------------------------------------------------------------+
# start of the script
# +-----------------------------------------------------------------------------------------------+

# save current directory
HDIR=`pwd`

# set installation versions
SHERPAVER="1.1.2"                                 # SHERPA version
HEPMC2VER="2.03.09"                               # HepMC2 version
LHAPDFVER="5.3.1"                                 # LHAPDF version

SHERPAWEBLOCATION=""                              # (web)location of SHERPA tarball
SHERPAFILE=""                                     # file name of SHERPA tarball

LVLCLEAN=0                                        # cleaning level (0-2)
FLGDEBUG="FALSE"                                  # debug flag for compilation

imode="LOCAL"                                     # operation mode (local installation/GRID)
# dummy setup (if all options are missing)
if [ "${imode}" = "LOCAL" ]; then
  instdir=${HDIR}                                 # installation directory for SHERPA
  patdir=${HDIR}                                  # location of required SHERPA patches
  cmsswd=${HDIR}/CMSSW_X_Y_Z                      # location of the CMSSW directory
  ship=${HDIR}                                    # location of SHERPA interface tarball
  shif=SherpaInterface.tgz                        # name of SHERPA interface tarball
  xopt="S_1.1.2@X_OpOfOF"                         # expert options
#  xopt=""
fi

if [ "${imode}" = "LOCAL" ]; then                 # local installation?
# get & evaluate options
  while getopts :i:p:d:l:f:o:m:W:S:C:Dh OPT
  do
    case $OPT in
    i) instdir=$OPTARG ;;
    p) patdir=$OPTARG ;;
    d) cmsswd=$OPTARG ;;
    l) ship=$OPTARG ;;
    f) shif=$OPTARG ;;
    o) xopt=$OPTARG ;;
    m) imode=$OPTARG ;;
    W) SHERPAWEBLOCATION=$OPTARG ;;
    S) SHERPAFILE=$OPTARG ;;
    C) LVLCLEAN=$OPTARG ;;
    D) FLGDEBUG=TRUE ;;
    h) print_help && exit 0 ;;
    \?)
      shift `expr $OPTIND - 1`
      if [ "$1" = "--help" ]; then print_help && exit 0;
      else 
        echo -n "SetupSherpaInterface: error: unrecognized option "
        if [ $OPTARG != "-" ]; then echo "'-$OPTARG'. try '-h'"
        else echo "'$1'. try '-h'"
        fi
        print_help && exit 1
      fi
      shift 1
      OPTIND=1
    esac
  done
fi

# make sure all path names are absolute
cd ${instdir}; instdir=`pwd`; cd ${HDIR}
cd ${patdir};  patdir=`pwd`;  cd ${HDIR}
if [ ! "${imode}" = "GRID" ]; then
  cd ${cmsswd};  cmsswd=`pwd`;  cd ${HDIR}
fi
if [ "${imode}" = "LOCAL" ]; then
  cd ${ship};    ship=`pwd`;    cd ${HDIR}
fi

echo " SHERPA interface setup (local/CRAB/GRID)"
echo "  -> SHERPA installation directory: '"${instdir}"'"
echo "  -> SHERPA version: '"${SHERPAVER}"'"
echo "  -> location of SHERPA patches/fixes: '"${patdir}"'"
if [ ! "${imode}" = "GRID" ]; then
  echo "  -> location of CMSSW: '"${cmsswd}"'"
  if [ "${imode}" = "LOCAL" ]; then
    echo "  -> location of SHERPA interface tarball: '"${ship}"'"
    echo "  -> name of SHERPA interface tarball: '"${shif}"'"
  fi
fi
echo "  -> expert options: '"${xopt}"'"
echo "  -> operation mode: '"${imode}"'"


echo "------------------------------------------------------------------------------------------"
echo " --> setup phase..."
echo "------------------------------------------------------------------------------------------"

# set path names for SHERPA interface in CMSSW
SHIFPTH1="GeneratorInterface"                     # subdirectory for the SHERPA interface
SHIFPTH2="SherpaInterface"                        # subdirectory for the SHERPA interface components
# set up file names
shshifile="install_sherpa.sh"                     # script for SHERPA installation
shhmifile="install_hepmc2.sh"                     # script for HepMC2 installation
shlhifile="install_lhapdf.sh"                     # script for LHAPDF installation
# XML TOOL definition files necessary for the SHERPA interface
toolshfile="sherpa.xml"
toolhmfile="hepmc.xml"
toollhfile="lhapdf.xml"

cd ${HDIR}
chmod u+x *.sh                                    # make scripts executable again

# set up paths and CMSSW properties
if [ "${imode}" = "LOCAL" ]; then
  SHERPAINTERFACELOCATION=${ship}                 # location SHERPA interface tarball
  SHERPAINTERFACEFILE=${shif}                     # name of SHERPA interface tarball
fi

# extract CMSSW version
if [ -e ${cmsswd} ]; then
  CMSSWVERM=`echo ${cmsswd} | awk 'match($0,/CMSSW_.*/){print substr($0,RSTART+6,1)}'`
  echo " --> recovered CMSSW version: "${CMSSWVERM}
  va=`echo ${CMSSWVERM} | cut -f1 -d"_"`
  vb=`echo ${CMSSWVERM} | cut -f2 -d"_"`
  vc=`echo ${CMSSWVERM} | cut -f3 -d"_"`
else
  if [ ! "${imode}" = "GRID" ]; then
    echo " <E> CMSSW directory does not exist: "${cmsswd}
    echo " <E> ...stopping..."
    exit 1
  fi
fi

# evaluate expert options
FORCESHERPA="FALSE"                               # flags to force (override) installation of SHERPA, HepMC2, LHAPDF
FORCEHEPMC2="FALSE"
FORCELHAPDF="FALSE"
OINST=""                                          # SHERPA installation options
#                                                 # ['p': SHERPA patches, 'h': HepMC2, 'l': LHAPDF, 'f': 32-bit comp. mode ]
# decode options
# S_version@H_version@L_version@X_(options)
chopt1="@"
chopt2="_"
xfcnt=`echo ${xopt} | grep -o ${chopt1} | grep -c ${chopt1}`
let xfcnt=${xfcnt}+1
lcnt=1
while [ ${lcnt} -le ${xfcnt} ]; do
  otmp=`echo ${xopt} | cut -f ${lcnt} -d ${chopt1}`
  otmpa=`echo ${otmp} | cut -f 1 -d ${chopt2}`
  otmpb=`echo ${otmp} | cut -f 2-99 -d ${chopt2}`
  if [ "${otmpa}" = "S" ]; then
    FORCESHERPA="TRUE"
    SHERPAVER=${otmpb}
  fi
  if [ "${otmpa}" = "H" ]; then
    FORCEHEPMC2="TRUE"
    FORCESHERPA="TRUE"
    HEPMC2VER=${otmpb}
  fi
  if [ "${otmpa}" = "L" ]; then
    FORCELHAPDF="TRUE"
    FORCESHERPA="TRUE"
    LHAPDFVER=${otmpb}
  fi
  if [ "${otmpa}" = "X" ]; then
    OINST=${otmpb}
  fi
  let lcnt=$lcnt+1
done
echo " <I> decoded overrides: S: "${FORCESHERPA}", H: "${FORCEHEPMC2}", L: "${FORCELHAPDF}
echo " <I> decoded versions: S: "${SHERPAVER}", H: "${HEPMC2VER}", L: "${LHAPDFVER}
echo " <I> decoded extra options: "${OINST}

if [ "${FORCESHERPA}" = "TRUE" ]; then
  MMTMP=`echo ${OINST} | grep -o "O" | grep -c "O"`
  lcnt=1
  OINST=""
  while [ ${lcnt} -le ${MMTMP} ]; do
    let lcnt=${lcnt}+1
    ctmp=`echo ${xopt} | cut -f ${lcnt} -d "O"`
    OINST=${OINST}""${ctmp}
  done
  echo "DEBUG: installation options: "$OINST
fi



echo "------------------------------------------------------------------------------------------"
echo " --> SHERPA installation phase..."
echo "------------------------------------------------------------------------------------------"

if [ ! -e  ${cmsswd} ]; then
  if [ ! "${imode}" = "GRID" ]; then
    echo " <E> CMSSW installation "${cmsswd}
    echo " <E>  does not exist -> stopping..."
    exit 1
  fi
else
# get SHERPA, HepMC2 and LHAPDF tool path in current CMSSW version
  cd ${cmsswd}
  export SHERPADIR=`scramv1 tool info sherpa  | grep -i sherpa_base  | cut -f2 -d"="`
  echo " <I> SHERPA directory in CMSSW is "${SHERPADIR}
  export HEPMC2DIR=`scramv1 tool info hepmc  | grep -i hepmc_base  | cut -f2 -d"="`
  echo " <I> HepMC2 directory in CMSSW is "${HEPMC2DIR}
  export LHAPDFDIR=`scramv1 tool info lhapdf | grep -i lhapdf_base | cut -f2 -d"="`
  export LHAPATH=${LHAPDFDIR} # needed since SHERPA 1.1.2
  echo " <I> LHAPDF directory in CMSSW is "${LHAPDFDIR}
  cd -
fi

# forced installation?
if [ "${FORCESHERPA}" = "TRUE" ]; then
  export SHERPADIR=${instdir}/SHERPA_${SHERPAVER} # SHERPA installation directory
  echo " <W> forcing SHERPA installation to path:"
  echo " <W> ... "${SHERPADIR}
fi
if [ "${FORCEHEPMC2}" = "TRUE" ]; then
  export HEPMC2DIR=${instdir}/HEPMC_${HEPMC2VER}    # HepMC2 installation directory
  echo " <W> forcing HepMC2 installation to path:"
  echo " <W> ... "${HEPMC2DIR}
fi
if [ "${FORCELHAPDF}" = "TRUE" ]; then
  export LHAPDFDIR=${instdir}/LHAPDF_${LHAPDFVER}   # LHAPDF installation directory
  echo " <W> forcing LHAPDF installation to path:"
  echo " <W> ... "${LHAPDFDIR}
fi


# forced SHERPA installation???
if [ "${FORCESHERPA}" = "TRUE" ]; then

# evaluate installation options
  ALLFLAGS=""
  ALLFLAGS=${ALLFLAGS}" -d "${instdir}
  ALLFLAGS=${ALLFLAGS}" -v "${SHERPAVER}
  ALLFLAGS=${ALLFLAGS}" -m "${HEPMC2VER}
  ALLFLAGS=${ALLFLAGS}" -l "${LHAPDFVER}" -L -l"
  ALLFLAGS=${ALLFLAGS}" -C "${LVLCLEAN}
#  ALLFLAGS=${ALLFLAGS}" -I "                                                                  # use configure/make/make install ?
  ALLFLAGS=${ALLFLAGS}" -P "                                                                  # softlink PDFsets ?
  ALLFLAGS=${ALLFLAGS}" -X "                                                                  # create XML files for CMSSW override
  if [ `echo ${OINST} | grep -c "p"` -gt 0 ]; then ALLFLAGS=${ALLFLAGS}" -p "${patdir};    fi # install SHERPA patches?
  if [ `echo ${OINST} | grep -c "F"` -gt 0 ]; then ALLFLAGS=${ALLFLAGS}" -F "${patdir};    fi # apply extra SHERPA fixes ?
  if [ `echo ${OINST} | grep -c "f"` -gt 0 ]; then ALLFLAGS=${ALLFLAGS}" -f";              fi # use 32-bit compatibility mode ?
  if [ `echo ${OINST} | grep -c "T"` -gt 0 ]; then ALLFLAGS=${ALLFLAGS}" -T";              fi # use multithreading ?
  if [ `echo ${OINST} | grep -c "Z"` -gt 0 ]; then ALLFLAGS=${ALLFLAGS}" -Z";              fi # use multiple cores ?
  if [ `echo ${OINST} | grep -c "K"` -gt 0 ]; then ALLFLAGS=${ALLFLAGS}" -K";              fi # keep source code ?
  if [ `echo ${OINST} | grep -c "A"` -gt 0 ]; then ALLFLAGS=${ALLFLAGS}" -A";              fi # enable analysis ?
  if [ "${FLGDEBUG}" = "TRUE" ];              then ALLFLAGS=${ALLFLAGS}" -D";              fi
###
  if [ ! "${SHERPAWEBLOCATION}" = "" ]; then ALLFLAGS=${ALLFLAGS}" -W "${SHERPAWEBLOCATION}; fi
  if [ ! "${SHERPAFILE}" = "" ];        then ALLFLAGS=${ALLFLAGS}" -S "${SHERPAFILE};      fi
###


# if needed, create installation directory
  if [ ! -d  ${instdir} ]; then
    echo " <W> installation directory does not exist, creating..."
    mkdir -p ${instdir}
  fi

# install SHERPA and - if required - HepMC2 and LHAPDF
  if [ ! -e ${SHERPADIR}/bin/Sherpa ]; then
    if [ -e  ${SHERPADIR} ]; then
      echo " <W> SHERPA directory exists but no executable found,"
      echo " <W>  deleting and reinstalling..."
      rm -rf ${SHERPADIR}
    fi
    echo " <I> installing SHERPA"
    echo ${HDIR}/${shshifile} ${ALLFLAGS}
    ${HDIR}/${shshifile} ${ALLFLAGS}
  else
    echo " <I> SHERPA already installed"
  fi

fi # check FORCESHERPA flag



echo "------------------------------------------------------------------------------------------"
echo " --> CMSSW + SHERPA setup phase..."
echo "------------------------------------------------------------------------------------------"

cd ${cmsswd}
xmldir=${cmsswd}/config/toolbox/slc4_ia32_gcc345/tools/selected
scramopt=""
if [ "${imode}" = "LOCAL" ]; then
  if [ "${FORCELHAPDF}" = "TRUE" ]; then # apply LHAPDF tool definition XML file
    toollhfile=`find ${HDIR} -type f -name lhapdf_*.xml`
    echo "XMLFILE(LHAPDF): "$toollhfile
#    cp ${HDIR}/${toollhfile} ${xmldir}/
#    scramv1 setup ${scramopt} lhapdf
  fi
  if [ "${FORCEHEPMC2}" = "TRUE" ]; then # apply HepMC tool definition XML file
    toolhmfile=`find ${HDIR} -type f -name hepmc_*.xml`
    echo "XMLFILE(HEPMC2): "$toolhmfile
#    cp ${HDIR}/${toolhmfile} ${xmldir}/
#    scramv1 setup ${scramopt} hepmc
  fi
  if [ "${FORCESHERPA}" = "TRUE" ]; then # apply SHERPA tool definition XML file
    toolshfile=`find ${HDIR} -type f -name sherpa_*.xml`
    echo "XMLFILE(SHERPA): "$toolshfile
#    cp ${HDIR}/${toolshfile} ${xmldir}/
#    scramv1 setup ${scramopt} sherpa
  fi
fi



echo "--------------------------------------------------------------------------------"
PDFDIR=`find ${LHAPATH} -name PDFsets`
echo " <I> if AFTER executing \"eval \`scramv1 ru -(c)sh\`\" LHAPATH is not defined "
echo " <I> please set environment variable LHAPATH to"
echo " <I>  "${PDFDIR}
echo " <I>  e.g. (BASH): export LHAPATH="${PDFDIR}
echo " <I>  e.g. (CSH):  setenv LHAPATH "${PDFDIR}
echo "--------------------------------------------------------------------------------"
