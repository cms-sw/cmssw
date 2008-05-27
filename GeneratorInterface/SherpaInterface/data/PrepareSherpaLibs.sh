#!/bin/bash
#
#  file:        PrepareSherpaLibs.sh
#  description: prepare SHERPA libraries and cross sections for local use
#               or for CRAB, generate CMSSW and CRAB config files & scripts
#               from templates
#  uses:        SHERPA datacards
#               Sherpa.cfg_template
#               crab.cfg_template
#               SetupSherpaInterface.sh
#
#  author:      Markus Merschmeyer, RWTH Aachen
#  date:        2008/05/13
#  version:     1.5
#



# +-----------------------------------------------------------------------------------------------+
# function definitions
# +-----------------------------------------------------------------------------------------------+

function print_help() {
    echo "" && \
    echo "PrepareSherpaLibs version 1.5" && echo && \
    echo "options: -i  path       path to SHERPA datacard, library & cross section files" && \
    echo "                         can also be in WWW (http://...) or SE (srm://...)" && \
    echo "                         -> ( "${datadir}" )" && \
    echo "         -p  process    SHERPA dataset/process name ( "${dataset}" )" && \
    echo "         -o  options    library/cross section options [LIBS,LBCR] ( "${dataopt}" )" && \
    echo "                         [ 'LIBS' : use libraries only               ]" && \
    echo "                         [ 'LBCR' : use libraries and cross sections ]" && \
    echo "         -D  filename   (optional) name of data card file     ( "${cfdc}" )" && \
    echo "         -L  filename   (optional) name of library file       ( "${cflb}" )" && \
    echo "         -C  filename   (optional) name of cross section file ( "${cfcr}" )" && \
    echo "         -d  path       path to CMSSW directory" && \
    echo "                         -> ( "${CMSSWDIR}" )" && \
    echo "         -m  mode       CMSSW running mode ['LOCAL','CRAB','cmsGen'] ( "${imode}" )" && \
    echo "         -n  SRM node   SE node for library transfer for CRAB mode" && \
    echo "                         -> ( "${MYSRMNAME}" )" && \
    echo "         -s  SRM path   SE path for library transfer for CRAB mode" && \
    echo "                         -> ( "${MYSRMPATH}" )" && \
    echo "         -c  path       (optional) path to CRAB directory" && \
    echo "                         -> ( "${CRABDIR}" )" && \
    echo "         -h             display this help and exit" && echo
}



### NEW function for preparing the SHERPA process dependent CMSSW include file
function build_cfi() {
# $1 : path to dataset (below CMSSW_X_Y_Z/src level)
# $2 : dataset (e.g. "em_ep_0j1incl")
  cfifile="sherpa_"$2".cfi"

  cat > ${cfifile}   <<EOF
source = SherpaSource
{
  untracked uint32 firstRun = 1

  untracked string libDir    = "SherpaRun" 
  untracked string resultDir = "Result" 

  PSet SherpaParameters = {

EOF

  echo "    vstring parameterSets = {"  >> ${cfifile}
  fcnt=0
  for file in `ls *.dat`; do
    let fcnt=${fcnt}+1
  done
  echo " <III> "${fcnt}" configuration files found for SHERPA"
  for file in `ls *.dat`; do
    let fcnt=${fcnt}-1
    if [ ${fcnt} -gt 0 ]; then
      echo ${file} | sed -e 's/.dat/\",/' | sed -e 's/^/\t\"/' >> ${cfifile}
    else
      echo ${file} | sed -e 's/.dat/\"/' | sed -e 's/^/\t\"/'  >> ${cfifile}
    fi
  done
  echo "    }" >> ${cfifile}
  echo ""      >> ${cfifile}

  for file in `ls *.dat`; do
    sed '/^$/d' < ${file} > ${file}.tmp        # remove empty lines
    lastline=`tail -n 1 ${file}.tmp` 
    datacard=`echo ${file}.tmp | sed s/.dat.tmp//`
    echo "vstring "${datacard}" = {" >> ${cfifile}
    cat  ${file}.tmp | sed s/"'"//g | sed 's/$/XXX/' | sed s/XXX/"',"/ | sed s/^/"'"/ >> ${cfifile}
    echo "'!'" >> ${cfifile}
    echo "}" >> ${cfifile}
    echo "" >> ${cfifile}
    rm ${file}.tmp
  done

  cat >> ${cfifile}   <<EOF
  }

}
EOF
}



# function to copy files from different locations (WWW, SE, local)
function file_copy() {
# $1 : target path
# $2 : file name
# $3 : destination path
#
#  srmopt=" -debug -streams_num=1 "
  srmopt=" -streams_num=1 "
#
  tpath="aaa/"
  fname="xxx.yyy"
  dpath="bbb/"
  if [ $# -ge 1 ]; then
    tpath=$1
    if [ $# -ge 2 ]; then
      fname=$2
      if [ $# -ge 3 ]; then
        dpath=$3
      fi
    fi
  fi
#
  if [ ! "${tpath}" = "${dpath}" ]; then
    if [ "${tpath}" = "./" ]; then
      tpath=${PWD}
    fi
    if [ "${dpath}" = "./" ]; then
      dpath=${PWD}
    fi
    cd /tmp
#
  TLOC0=`echo ${tpath} | cut -f1 -d "/" | grep -c -i http`
  TLOC1=`echo ${tpath} | cut -f1 -d "/" | grep -c -i srm`
  if [ ${TLOC0} -gt 0 ]; then    # file is in WWW
    echo " <I> retrieving WWW file: "${fname}
    echo " <I>  from path: "${tpath}
    wget ${tpath}/${fname}
  elif [ ${TLOC1} -gt 0 ]; then  # file is in SE
    echo " <I> retrieving SE file: "${fname}
    echo " <I>  from path: "${tpath}
    srmcp ${srmopt} ${tpath}/${fname} file:///${PWD}/${fname}
  else                           # local file
    echo " <I> copying local file: "${fname}
    echo " <I>  from path: "${tpath}
    cp ${tpath}/${fname} ./
  fi
#
  DLOC0=`echo ${dpath} | cut -f1 -d "/" | grep -c -i http`
  DLOC1=`echo ${dpath} | cut -f1 -d "/" | grep -c -i srm`
  if [ ${DLOC1} -gt 0 ]; then    # file has to go to SE
    echo " <I> storing SE file: "${fname}
    echo " <I>  to path: "${dpath}
    srmcp ${srmopt} file:///${PWD}/${fname} ${dpath}/${fname}
  else                           # local file
    echo " <I> copying local file: "${fname}
    echo " <I>  to path: "${dpath}
    cp ${fname} ${dpath}/
  fi
#
    rm ${fname}
    cd -
  fi
}



# +-----------------------------------------------------------------------------------------------+
# start of the script
# +-----------------------------------------------------------------------------------------------+

# save current path name
HDIR=`pwd`

# 'guess' SHERPA dataset name
ip=1
cproc="XXX"
for file in `ls sherpa_*_libs.tgz`; do
  if [ $ip -eq 1 ]; then
    cproc=`echo $file | awk -F"sherpa_" '{print $2}' | awk -F"_libs" '{print $1}'`
  fi
  let ip=$ip+1
done

# dummy setup (if all options are missing)
datadir=${HDIR}                                      # path to SHERPA datacards (libraries)
#dataset="XXX"                                        # SHERPA dataset/process name
dataset=${cproc}                                     # SHERPA dataset/process name
dataopt="LBCR"                                       # library/cross section option
cfdc=""                                              # custom data card file name
cflb=""                                              # custom library file name
cfcr=""                                              # custom cross section file name
CMSSWVER=2_0_6                                       # CMSSW version
CMSSWDIR=${HDIR}/CMSSW_${CMSSWVER}                   # CMSSW directory
#CMSSWDIR=${HOME}/CMSSW_${CMSSWVER}                   # CMSSW directory
#CMSSWDIR=${HOME}/scratch0/CMSSW_${CMSSWVER}          # CMSSW directory
CRABVER=2_1_2                                        # CRAB version
CRABDIR=${HOME}/CRAB_${CRABVER}                      # CRAB directory
MYSRMNAME="grid-srm.physik.rwth-aachen.de"           # name of SRM node
MYSRMPATH="/pnfs/physik.rwth-aachen.de/dcms/merschm" # SRM path to SHERPA library & cross section tarballs
imode="LOCAL"                                        # CMSSW running mode
###
MYPATCHES=${HDIR}                                    # local path to SHERPA patches
MYSCRIPTS=${HDIR}                                    # local path to shell scripts
###

# get & evaluate options
while getopts :i:p:o:D:L:C:d:c:n:s:m:h OPT
do
  case $OPT in
  i) datadir=$OPTARG ;;
  p) dataset=$OPTARG ;;
  o) dataopt=$OPTARG ;;
  D) cfdc=$OPTARG ;;
  L) cflb=$OPTARG ;;
  C) cfcr=$OPTARG ;;
  d) CMSSWDIR=$OPTARG ;;
  c) CRABDIR=$OPTARG ;;
  n) MYSRMNAME=$OPTARG ;;
  s) MYSRMPATH=$OPTARG ;;
  m) imode=$OPTARG ;;
  h) print_help && exit 0 ;;
  \?)
    shift `expr $OPTIND - 1`
    if [ "$1" = "--help" ]; then print_help && exit 0;
    else 
      echo -n "PrepareSherpaLibs: error: unrecognized option "
      if [ $OPTARG != "-" ]; then echo "'-$OPTARG'. try '-h'"
      else echo "'$1'. try '-h'"
      fi
      print_help && exit 1
    fi
    shift 1
    OPTIND=1
  esac
done

# make sure to use absolute path names...
xpth=`echo ${datadir} | cut -f1 -d"/"`
if [ "$xpth" = "" ] || [ "$xpth" = "." ] || [ "$xpth" = ".." ] || [ "$xpth" = "~" ]; then
  echo " <I> fixing data directory: "${datadir}
  cd ${datadir};  datadir=`pwd`;  cd ${HDIR}
  echo " <I>    to: "${datadir}
fi
cd ${CMSSWDIR}; CMSSWDIR=`pwd`; cd ${HDIR}
if [ "${imode}" = "CRAB" ]; then
  cd ${CRABDIR};  CRABDIR=`pwd`;  cd ${HDIR}
fi


echo "  -> data card directory '"${datadir}"'"
echo "  -> dataset name '"${dataset}"'"
echo "  -> library & cross section otions '"${dataopt}"'"
echo "  -> CMSSW directory '"${CMSSWDIR}"'"
echo "  -> operation mode: '"${imode}"'"
if [ "${imode}" = "CRAB" ]; then
  echo "  -> CRAB directory '"${CRABDIR}"'"
  echo "  -> SRM node: '"${MYSRMNAME}"'"
  echo "  -> SRM path: '"${MYSRMPATH}"'"
fi


# set up 
SHIFPTH="GeneratorInterface/SherpaInterface/data"             # path to 'data' directory of SHERPA interface in CMSSW
SHIFPTT="GeneratorInterface/SherpaInterface/test"             # path to 'test' directory of SHERPA interface in CMSSW
SHIFRUN="SherpaRun"                                           # directory name for SHERPA process related stuff
MYCMSSWDATA=${CMSSWDIR}/src/${SHIFPTH}                        # local path to 'data' directory of SHERPA interface in CMSSW
MYCMSSWTEST=${CMSSWDIR}/src/${SHIFPTT}                        # local path to 'test' directory of SHERPA interface in CMSSW
SHIFPTH_DAT=${MYCMSSWTEST}/${SHIFRUN}                         # local paths for SHERPA process related stuff (libs, cross s.)
if [ "${cfdc}" = "" ]; then
  cardfile=sherpa_${dataset}_cards.tgz                        # set SHERPA data file names
else
  cardfile=${cfdc}
fi
if [ "${cflb}" = "" ]; then
  libsfile=sherpa_${dataset}_libs.tgz
else
  libsfile=${cflb}
fi
if [ "${cfcr}" = "" ]; then
  crssfile=sherpa_${dataset}_crss.tgz
else
  crssfile=${cfcr}
fi
SHERPATEMPLATE=${MYCMSSWDATA}/Sherpa.cfg_template             # name of template file for SHERPA interface configuration
SHERPACONFIG="Sherpa.cfg"                                     # name of SHERPA interface configuration file
CRABTMPFILE1="crab_1.tmp"                                     # temporary files
CRABTMPFILE2="crab_2.tmp"
CRABTEMPLATE=${MYCMSSWDATA}/crab.cfg_template                 # name of template file for CRAB config file
CRABCONFIG="crab.cfg"                                         # name of CRAB config file
INSTTEMPLATE="SetupSherpaInterface.sh"                        # name of template file for setup script
MYCRABSCRIPT="crab_sherpa.sh"                                 # name of setup script for SHERPA interface for CRAB



# test/initialize voms-proxy
if [ "${imode}" = "CRAB" ]; then
  infovoms="voms-proxy-info"                                  # command to get VOMS proxy information
  initvoms="voms-proxy-init"                                  # command to initialize VOMS proxy
  myvoms="cms:/cms/dcms"                                      # your Virtual Organization Membership (cms, dcms ?)
  if [ `${infovoms} | grep -c -i timeleft` -eq 0 ]; then
    echo " <I> no active voms proxy found, trying to create a new one..."
    echo -n " <I>  specify your VOMS server (e.g. "${myvoms}"): "
    read tmpvoms
    if [ ! "${tmpvoms}" = "" ]; then
      myvoms=${tmpvoms}
    fi
    ${initvoms} -voms ${myvoms} -valid 192:00
  else
    rtime=`${infovoms} | grep -i timeleft | cut -f4 -d" "`
    echo " <I> active voms proxy found, remaining time: "${rtime}
  fi
fi



###
### SHERPA part
###

if [ "${imode}" = "cmsGen" ];then
  MYCMSSWDATA=${CMSSWDIR}/../
  MYCMSSWTEST=${CMSSWDIR}/../
  SHIFPTH_DAT=${MYCMSSWTEST}/${SHIFRUN}
  SHERPATEMPLATE=${MYCMSSWDATA}/Sherpa.cfg_template
  CRABTEMPLATE=${MYCMSSWDATA}/crab.cfg_template
  mkdir ${MYCMSSWTEST}/${SHIFRUN} 
fi


if [ -e ${CMSSWDIR} ]; then
  CMSSWVER=`echo ${CMSSWDIR} | awk 'match($0,/CMSSW_.*/){print substr($0,RSTART+6,RLENGTH-6)}'`
  echo " --> recovered CMSSW version: "${CMSSWVER}
else
  echo " <E> CMSSW directory does not exist: "${CMSSWDIR}
  echo " <E> ...stopping..."
  exit 1
fi

# create dataset directory tree
cd ${MYCMSSWDATA}
if [ -e ${SHIFPTH_DAT} ]; then
  echo " <W> dataset directory "${SHIFPTH_DAT}" exists"
  echo " <W> ...removing..."
  rm -rf ${SHIFPTH_DAT}
fi
mkdir ${SHIFPTH_DAT}

# get & unpack dataset files, generate .cff and .cfi files
cd ${SHIFPTH_DAT}

 file_copy ${datadir} ${cardfile} ${PWD}
 tar -xzf ${cardfile}; rm ${cardfile}
 build_cfi ${MYCMSSWTEST}/${SHIFRUN} ${dataset}

 if [ "${imode}" = "LOCAL" ]; then
   dataloc=${PWD}
 elif [ "${imode}" = "CRAB" ]; then
   dataloc="srm://"${MYSRMNAME}":8443/"${MYSRMPATH}
 elif [ "${imode}" = "cmsGen" ]; then
   dataloc=${PWD}
 fi

 file_copy ${datadir} ${libsfile} ${dataloc}
 if [ "${dataopt}" = "LBCR" ]; then
   file_copy ${datadir} ${crssfile} ${dataloc}
 fi

 if [ ! "${imode}" = "CRAB" ]; then
   tar -xzf ${libsfile}; rm ${libsfile}
   if [ "${dataopt}" = "LBCR" ]; then
     tar -xzf ${crssfile}; rm ${crssfile}
   fi
 fi
cd -


# produce CMSSW .cfg file from template
cd ${MYCMSSWTEST}
cp ${MYCMSSWTEST}/${SHIFRUN}/sherpa_${dataset}.cfi ${PWD}
if [ ! "${imode}" = "cmsGen" ]; then
  sed -e 's:MYSHERPAPROCESS:'${dataset}':' < ${SHERPATEMPLATE} > ${SHERPACONFIG}
fi

# display information
echo " <I> the SHERPA interface has been set up:"
echo " <I>   CMSSW version: "${CMSSWVER}
echo " <I>   CMSSW path: "${CMSSWDIR}
echo " <I>   data set (process): "${dataset}
echo " <I>    -> taken from: "${datadir}
echo " <I>   data set option: "${dataopt}
echo " <I>   generated configuration files:"
echo " <I>    -> "${dataset}".cfi"
echo " <I>    -> Sherpa.cfg"



if [ "${imode}" = "CRAB" ]; then

###
### CRAB part
###

  if [ -e ${CRABDIR} ]; then
    CRABVER=`echo ${CRABDIR} | awk 'match($0,/CRAB_.*/){print substr($0,RSTART+5,RLENGTH-5)}'`
    echo " --> recovered CRAB version: "${CRABVER}
  else
    echo " <E> CRAB directory does not exist: "${CRABDIR}
    echo " <E> ...stopping..."
    exit 1
  fi

# create CRAB .cfg file and setup script from templates
  cd ${MYCMSSWTEST}

  cp ${CRABTEMPLATE} ${CRABTMPFILE1}
   sed -e 's:MYCRABSCRIPT:'${MYCRABSCRIPT}':' < ${CRABTMPFILE1} > ${CRABTMPFILE2}
   mv ${CRABTMPFILE2} ${CRABTMPFILE1}
   while [ `grep -c MYCMSSWTEST ${CRABTMPFILE1}` -gt 0 ]; do
     sed -e 's:MYCMSSWTEST:'${MYCMSSWTEST}':' < ${CRABTMPFILE1} > ${CRABTMPFILE2}
     mv ${CRABTMPFILE2} ${CRABTMPFILE1}
   done
   while [ `grep -c MYCMSSWDATA ${CRABTMPFILE1}` -gt 0 ]; do
     sed -e 's:MYCMSSWDATA:'${MYCMSSWDATA}':' < ${CRABTMPFILE1} > ${CRABTMPFILE2}
     mv ${CRABTMPFILE2} ${CRABTMPFILE1}
   done
   sed -e 's:MYSHERPAPROCESS:'${dataset}':' < ${CRABTMPFILE1} > ${CRABTMPFILE2}
   mv ${CRABTMPFILE2} ${CRABTMPFILE1}
   sed -e 's:MYPATCHES:'${MYPATCHES}':' < ${CRABTMPFILE1} > ${CRABTMPFILE2}
   mv ${CRABTMPFILE2} ${CRABTMPFILE1}
   sed -e 's:MYSCRIPTS:'${MYSCRIPTS}':' < ${CRABTMPFILE1} > ${CRABTMPFILE2}
   mv ${CRABTMPFILE2} ${CRABTMPFILE1}
   sed -e 's:MYSRMNAME:'${MYSRMNAME}':' < ${CRABTMPFILE1} > ${CRABTMPFILE2}
   mv ${CRABTMPFILE2} ${CRABTMPFILE1}
   sed -e 's:MYSRMPATH:'${MYSRMPATH}':' < ${CRABTMPFILE1} > ${CRABTMPFILE2}
   mv ${CRABTMPFILE2} ${CRABTMPFILE1}
  mv ${CRABTMPFILE1} ${CRABCONFIG}

  cp ${MYSCRIPTS}/${INSTTEMPLATE} ${CRABTMPFILE1}
   sed -e 's:imode="LOCAL":imode="CRAB":' < ${CRABTMPFILE1} > ${CRABTMPFILE2}
   mv ${CRABTMPFILE2} ${CRABTMPFILE1}
   sed -e 's|dataloc="XXX"|dataloc='${dataloc}'|' < ${CRABTMPFILE1} > ${CRABTMPFILE2}
   mv ${CRABTMPFILE2} ${CRABTMPFILE1}
   sed -e 's:dataset="YYY":dataset='${dataset}':' < ${CRABTMPFILE1} > ${CRABTMPFILE2}
   mv ${CRABTMPFILE2} ${CRABTMPFILE1}
  mv ${CRABTMPFILE1} ${MYCRABSCRIPT}
  chmod u+x ${MYCRABSCRIPT}


# display information
  echo " <I> the SHERPA interface has been set up:"
  echo " <I>   CRAB version: "${CRABVER}
  echo " <I>   CRAB path: "${CRABDIR}
  echo " <I>   generated configuration file:"
  echo " <I>    -> "${CRABCONFIG}
  echo " <I>   generated shell script:"
  echo " <I>    -> "${MYCRABSCRIPT}


fi # check imode flag (=CRAB?)


cd ${HDIR}
