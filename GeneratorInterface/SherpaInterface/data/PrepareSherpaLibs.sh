#!/bin/bash
#
#  file:        PrepareSherpaLibs.sh
#  description: prepare SHERPA libraries and cross sections for local use
#               generate CMSSW python script
#  uses:        SHERPA datacards, libs and cross sections
#
#  author:      Markus Merschmeyer, RWTH Aachen
#  date:        2008/10/27
#  version:     2.2
#



# +-----------------------------------------------------------------------------------------------+
# function definitions
# +-----------------------------------------------------------------------------------------------+

function print_help() {
    echo "" && \
    echo "PrepareSherpaLibs version 2.2" && echo && \
    echo "options: -i  path       path to SHERPA datacard, library & cross section files" && \
    echo "                         can also be in WWW (http://...) or SE (srm://...)" && \
    echo "                         -> ( "${datadir}" )" && \
    echo "         -p  process    SHERPA dataset/process name ( "${dataset}" )" && \
    echo "         -o  options    library/cross section options [LIBS,LBCR] ( "${dataopt}" )" && \
    echo "                         [ 'LIBS' : use libraries only               ]" && \
    echo "                         [ 'LBCR' : use libraries and cross sections ]" && \
    echo "         -d  path       path to CMSSW directory" && \
    echo "                         -> ( "${CMSSWDIR}" )" && \
    echo "         -m  mode       CMSSW running mode ['LOCAL','CRAB','cmsGen'] ( "${imode}" )" && \
    echo "         -a  path       user analysis path inside CMSSW ( "${MYANADIR}" )" && \
    echo "         -D  filename   (optional) name of data card file     ( "${cfdc}" )" && \
    echo "         -L  filename   (optional) name of library file       ( "${cflb}" )" && \
    echo "         -C  filename   (optional) name of cross section file ( "${cfcr}" )" && \
    echo "         -h             display this help and exit" && echo
}


# function to build a python script for cmsDriver
function build_python_cfi() {
cat > sherpa_cfi.py << EOF
import FWCore.ParameterSet.Config as cms

source=cms.Source("SherpaSource",
  firstRun  = cms.untracked.uint32(1),
  libDir    = cms.untracked.string('SherpaRun'),
  resultDir = cms.untracked.string('Result')
)
EOF
}


# function to build a python script for cmsRun
function build_python_cfg() {
cat > sherpa_cfg.py << EOF
import FWCore.ParameterSet.Config as cms

process = cms.Process("runSherpa")
process.source=cms.Source("SherpaSource",
  firstRun  = cms.untracked.uint32(1),
  libDir    = cms.untracked.string('SherpaRun'),
  resultDir = cms.untracked.string('Result')
)
process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    sourceSeed = cms.untracked.uint32(98765)
)
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)
process.sherpa_out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('sherpa_GEN.root')
)
process.outpath = cms.EndPath(process.sherpa_out)
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
dataset=${cproc}                                     # SHERPA dataset/process name
dataopt="LBCR"                                       # library/cross section option
CMSSWDIR=${HDIR}/CMSSW_X_Y_Z                         # CMSSW directory
imode="LOCAL"                                        # CMSSW running mode
MYANADIR="GeneratorInterface/SherpaInterface"        # user analysis directory inside CMSSW
#                                                    # -> CMSSW_X_Y_Z/src/${MYANADIR}/
cfdc=""                                              # custom data card file name
cflb=""                                              # custom library file name
cfcr=""                                              # custom cross section file name

# get & evaluate options
while getopts :i:p:o:d:m:a:D:L:C:h OPT
do
  case $OPT in
  i) datadir=$OPTARG ;;
  p) dataset=$OPTARG ;;
  o) dataopt=$OPTARG ;;
  d) CMSSWDIR=$OPTARG ;;
  m) imode=$OPTARG ;;
  a) MYANADIR=$OPTARG ;;
  D) cfdc=$OPTARG ;;
  L) cflb=$OPTARG ;;
  C) cfcr=$OPTARG ;;
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
  cd ${datadir} && datadir=`pwd`;  cd ${HDIR}
  echo " <I>    to: "${datadir}
fi
cd ${CMSSWDIR} && CMSSWDIR=`pwd`; cd ${HDIR}
if [ ! -e ${CMSSWDIR} ]; then                                 # check for CMSSW directory
  echo " <E> CMSSW directory does not exist: "${CMSSWDIR}
  echo " <E> ...stopping..."
  exit 1
fi

# set up various (path) names
MYCMSSWTEST=${CMSSWDIR}/src/${MYANADIR}/test                  # local path to 'test' directory of SHERPA interface in CMSSW
MYCMSSWSHPA=${MYCMSSWTEST}/SherpaRun                          # local paths for SHERPA process related stuff (.dat,libs,cross s.)

# print current options/parameters
echo "  -> data card directory '"${datadir}"'"
echo "  -> dataset name '"${dataset}"'"
echo "  -> library & cross section otions '"${dataopt}"'"
echo "  -> CMSSW directory '"${CMSSWDIR}"'"
echo "  -> operation mode: '"${imode}"'"
echo "  -> CMSSW user analysis path: '"${MYANADIR}"'"

# set up 
if [ ! -e ${CMSSWDIR}/src/${MYANADIR}/test ]; then            # create user analysis path
  echo " <W> CMSSW user analysis path "${MYANADIR}/test" does not exist,..."
  echo " <W> ...creating"
  mkdir -p ${CMSSWDIR}/src/${MYANADIR}/test
else
  rm ${CMSSWDIR}/python/${MYANADIR}/*.py*                     # ...clean up
fi
if [ ! -e ${CMSSWDIR}/src/${MYANADIR}/python ]; then          # create 'python' subdirectory
  mkdir -p ${CMSSWDIR}/src/${MYANADIR}/python
else
  rm ${CMSSWDIR}/src/${MYANADIR}/python/*.py*                 # ...clean up
fi

if [ "${cfdc}" = "" ]; then
  cardfile=sherpa_${dataset}_crdE.tgz                         # set SHERPA data file names
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

###
### SHERPA part
###

# create dataset directory tree
if [ -e ${MYCMSSWSHPA} ]; then
  echo " <W> dataset directory "${MYCMSSWSHPA}" exists"
  echo " <W> ...removing..."
  rm -rf ${MYCMSSWSHPA}
fi
mkdir -p ${MYCMSSWSHPA}

# get & unpack dataset files, generate .cff and .cfi files
cd ${MYCMSSWSHPA}
 file_copy ${datadir} ${cardfile} ${PWD}
 tar -xzf ${cardfile}; rm ${cardfile}
 file_copy ${datadir} ${libsfile} ${PWD}
 tar -xzf ${libsfile}; rm ${libsfile}
 if [ "${dataopt}" = "LBCR" ]; then
   file_copy ${datadir} ${crssfile} ${PWD}
   tar -xzf ${crssfile}; rm ${crssfile}
 fi
cd -

# generate & compile pyhton script
cd ${MYCMSSWTEST}
build_python_cfi
mv *_cfi.py ../python/
build_python_cfg
cd ..
scramv1 b
cd -

cd ${HDIR}
