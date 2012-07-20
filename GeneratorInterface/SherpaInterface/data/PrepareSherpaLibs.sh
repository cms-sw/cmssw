#!/bin/bash
#
#  file:        PrepareSherpaLibs.sh
#  description: prepare SHERPA libraries and cross sections for local use
#               generate CMSSW python script
#  uses:        SHERPA datacards, libs and cross sections
#
#  author:      Markus Merschmeyer, Sebastian Thuer, RWTH Aachen
#  date:        17th July 2012
#  version:     4.0



# +-----------------------------------------------------------------------------------------------+
# function definitions
# +-----------------------------------------------------------------------------------------------+

function print_help() {
    echo "" && \
    echo "PrepareSherpaLibs version 4.0" && echo && \
    echo "options: -i  path       path to SHERPA datacard, library & cross section files" && \
    echo "                         can also be in WWW (http://...) or SE (srm://...)" && \
    echo "                         -> ( "${datadir}" )" && \
    echo "         -p  process    SHERPA dataset/process name ( "${dataset}" )" && \
    echo "         -m  mode       CMSSW running mode ( "${imode}" )" && \
    echo "                         [ 'LOCAL'  : local running of CMSSW         ]" && \
    echo "                         [ 'PROD'   : for production validation      ]" && \
    echo "         -c  condition  running conditions ( "${MYCONDITIONS}" )" && \
    echo "         -a  path       user analysis path inside CMSSW ( "${MYANADIR}" )" && \
    echo "         -D  filename   (optional) name of data card file     ( "${cfdc}" )" && \
    echo "         -L  filename   (optional) name of library file       ( "${cflb}" )" && \
    echo "         -C  filename   (optional) name of cross section file ( "${cfcr}" )" && \
    echo "         -G  filename   (optional) name of MI grid file       ( "${cfgr}" )" && \
    echo "         -P  SRM path   (CRAB) SE path for final results" && \
    echo "                         -> ( "${MYSRMPATH}" )" && \
    echo "         -h             display this help and exit" && echo
}




# function to build a python script for cmsDriver
function build_python_cfi() {

  shpacfifile=$1  # config file name
  process=$2      # process name
  checksum=$3     # md5 checksum

  if [ -e ${shpacfifile} ]; then rm ${shpacfifile}; fi
  touch ${shpacfifile}

  echo "import FWCore.ParameterSet.Config as cms"                          >> ${shpacfifile}
  echo "import os"                                                         >> ${shpacfifile} 
  echo ""                                                                  >> ${shpacfifile}
  echo "source = cms.Source(\"EmptySource\")"                              >> ${shpacfifile}
  echo ""                                                                  >> ${shpacfifile}
  echo "generator = cms.EDFilter(\"SherpaGeneratorFilter\","               >> ${shpacfifile}
  echo "  maxEventsToPrint = cms.untracked.int32(0),"                      >> ${shpacfifile}
  echo "  filterEfficiency = cms.untracked.double(1.0),"                   >> ${shpacfifile}
  echo "  crossSection = cms.untracked.double(-1),"                        >> ${shpacfifile}
  echo "  SherpaProcess = cms.untracked.string('"${process}"'),"           >> ${shpacfifile}
  echo "  SherpaChecksum = cms.untracked.string('"${checksum}"'),"         >> ${shpacfifile}
  echo "  SherpaPath = cms.untracked.string(os.getcwd()),"                 >> ${shpacfifile}
  echo "  SherpaPathPiece = cms.untracked.string(os.getcwd()),"            >> ${shpacfifile}
  echo "  SherpaResultDir = cms.untracked.string('Result'),"               >> ${shpacfifile}
  echo "  SherpaDefaultWeight = cms.untracked.double(1.0),"                >> ${shpacfifile}
  echo "  SherpaParameters = cms.PSet(parameterSets = cms.vstring("        >> ${shpacfifile}
  fcnt=0
  for file in `ls *.dat`; do
    let fcnt=${fcnt}+1
  done
  for file in `ls *.dat`; do
    let fcnt=${fcnt}-1
    pstnam=`echo ${file} | cut -f1 -d"."`
    if [ ${fcnt} -gt 0 ]; then
  echo "                             \""${pstnam}"\","    >> ${shpacfifile}
    else
  echo "                             \""${pstnam}"\"),"   >> ${shpacfifile}
    fi
  done
  for file in `ls *.dat`; do
    pstnam=`echo ${file} | cut -f1 -d"."`
  echo "                              "${pstnam}" = cms.vstring(" >> ${shpacfifile}
    cp ${file} ${file}.tmp1
    sed -e 's/[%\!].*//g' < ${file}.tmp1 > ${file}.tmp2            # remove comment lines (beginning with % or !)
    mv ${file}.tmp2 ${file}.tmp1
    sed -e 's/^[ \t]*//;s/[ \t]*$//' < ${file}.tmp1 > ${file}.tmp2 # remove beginnig & trailing whitespaces
    mv ${file}.tmp2 ${file}.tmp1
    sed '/^$/d' < ${file}.tmp1 > ${file}.tmp2                      # remove empty lines
    mv ${file}.tmp2 ${file}.tmp1
    sed -e 's/^/ /g;s/ (/(/;s/ }/}/' < ${file}.tmp1 > ${file}.tmp2 # add single space in front of parameters
    mv ${file}.tmp2 ${file}.tmp1
###
    sed -e 's/\"/\\"/g' < ${file}.tmp1 > ${file}.tmp2              # protect existing '"' by '\"'
    mv ${file}.tmp2 ${file}.tmp1
###
    sed -e 's/^/\t\t\t\t"/;s/$/\",/' < ${file}.tmp1 > ${file}.tmp2 # add ([]") and ("') around parameters
    mv ${file}.tmp2 ${file}.tmp1
    sed -e '$s/\",/\"/' < ${file}.tmp1 > ${file}.tmp2              # fix last line
    mv ${file}.tmp2 ${file}.tmp1
    cat  ${file}.tmp1                                         >> ${shpacfifile}
  echo "                                                  )," >> ${shpacfifile}
    rm ${file}.tmp*
  done
  echo "                             )"                       >> ${shpacfifile}
  echo ")"                                                    >> ${shpacfifile}
  echo ""                                                     >> ${shpacfifile}
#  echo "ProducerSourceSequence = cms.Sequence(generator)"     >> ${shpacfifile}
  echo "ProductionFilterSequence = cms.Sequence(generator)"   >> ${shpacfifile}
  echo ""                                                     >> ${shpacfifile}

#  cat > sherpa_custom_cff.py << EOF
#import FWCore.ParameterSet.Config as cms
#
#def customise(process):
#
#	process.genParticles.abortOnUnknownPDGCode = False
#
#	return(process)
#EOF

}


# function to build a python script for cmsRun
function build_python_cfg() {

  shpacfgfile=$1  # config file name
  shpaoutfile=$2  # output root file name
  shpacfifile=$3  # include file name (for source)
  shpacfifile=`echo ${shpacfifile} | sed -e 's/\.py//'`
  shpacfipath=$4  # include file path (for source)

  if [ -e ${shpacfgfile} ]; then rm ${shpacfgfile}; fi
  touch ${shpacfgfile}

  echo "import FWCore.ParameterSet.Config as cms"                                                  >> ${shpacfgfile}
  echo ""                                                                                          >> ${shpacfgfile}
  echo "process = cms.Process(\"runSherpa\")"                                                      >> ${shpacfgfile}
  echo "process.load('${shpacfipath}/${shpacfifile}')"                                             >> ${shpacfgfile}
  echo "process.load(\"Configuration.StandardSequences.SimulationRandomNumberGeneratorSeeds_cff\")">> ${shpacfgfile}
  echo "process.maxEvents = cms.untracked.PSet("                                                   >> ${shpacfgfile}
  echo "    input = cms.untracked.int32(100)"                                                      >> ${shpacfgfile}
  echo ")"                                                                                         >> ${shpacfgfile}
  echo "process.randomEngineStateProducer = cms.EDProducer(\"RandomEngineStateProducer\")"	   >> ${shpacfgfile}
  echo "process.p1 = cms.Path(process.randomEngineStateProducer)"				   >> ${shpacfgfile}
  echo "process.path = cms.Path(process.generator)"					           >> ${shpacfgfile}
  echo "process.GEN = cms.OutputModule(\"PoolOutputModule\","                                      >> ${shpacfgfile}
  echo "    fileName = cms.untracked.string('"${shpaoutfile}"')"                                   >> ${shpacfgfile}
  echo ")"                                                                                         >> ${shpacfgfile}
  echo "process.outpath = cms.EndPath(process.GEN)"                                                >> ${shpacfgfile}
#  echo ""                                                                                         >> ${shpacfgfile}
#  echo "process.genParticles.abortOnUnknownPDGCode = False"                                       >> ${shpacfgfile}
  echo "process.schedule = cms.Schedule(process.p1,process.path,process.outpath)"                  >> ${shpacfgfile}
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
    rm -f ${fname}
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
if [ -e ${CMSSW_BASE} ]; then
  CMSSWDIR=${CMSSW_BASE}                             # CMSSW directory
else
  if [ ! "${imode}" = "PROD" ];then
    echo " <E> \$CMSSW_BASE "${CMSSW_BASE}" does not exist"
    echo " <E> stopping..."
    exit 1
  fi
fi
imode="PROD"                                         # CMSSW running mode
MYANADIR="A/B"                                       # user analysis directory inside CMSSW
#                                                    # -> CMSSW_X_Y_Z/src/${MYANADIR}/
cfdc=""                                              # custom data card file name
cflb=""                                              # custom library file name
cfcr=""                                              # custom cross section file name
cfgr=""                                              # custom MI grid file name
MYSRMPATH="./"                                       # SRM path for storage of results
#MYCONDITIONS="IDEAL_V11"                             # CMSSW_2_2_X conditions
MYCONDITIONS="MC_31X_V5"                             # CMSSW_3_1/2_X conditions
#MYCONDITIONS="STARTUP31X_V4"                         # CMSSW_3_1/2_X conditions
#MYCONDITIONS="DESIGN_31X_V4"                         # CMSSW_3_1/2_X conditions
TDIR=TMP


# get & evaluate options
while getopts :i:p:d:m:c:a:D:L:C:G:P:h OPT
do
  case $OPT in
  i) datadir=$OPTARG ;;
  p) dataset=$OPTARG ;;
  m) imode=$OPTARG ;;
  c) MYCONDITIONS=$OPTARG ;;
  a) MYANADIR=$OPTARG ;;
  D) cfdc=$OPTARG ;;
  L) cflb=$OPTARG ;;
  C) cfcr=$OPTARG ;;
  G) cfgr=$OPTARG ;;
  P) MYSRMPATH=$OPTARG ;;
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

# print current options/parameters
echo "  -> data card directory '"${datadir}"'"
echo "  -> dataset name '"${dataset}"'"
echo "  -> operation mode: '"${imode}"'"
echo "  -> running conditions: '"${MYCONDITIONS}"'"
echo "  -> CMSSW user analysis path: '"${MYANADIR}"'"


# set up 
if [ "${imode}" = "PROD" ] || [ "${imode}" = "GRID" ]; then
  MYCMSSWTEST=${HDIR}/${TDIR}
  MYCMSSWPYTH=${HDIR}/${TDIR}
  MYCMSSWSHPA=${HDIR}/${TDIR}
else
  MYCMSSWTEST=${CMSSWDIR}/src/${MYANADIR}/test
  MYCMSSWPYTH=${CMSSWDIR}/src/${MYANADIR}/python
  MYCMSSWSHPA=${CMSSWDIR}/src/${MYANADIR}/test
  if [ ! -e ${MYCMSSWTEST} ]; then                            # create user analysis path
    mkdir -p ${MYCMSSWTEST}
  else
    rm -f ${CMSSWDIR}/python/${MYANADIR}/*.py*                # ...clean up
  fi
  if [ ! -e ${MYCMSSWPYTH} ]; then                            # create 'python' subdirectory
    mkdir -p ${MYCMSSWPYTH}
  else
    rm -f ${MYCMSSWPYTH}/*.py*                                # ...clean up
  fi
fi


# set SHERPA data file names
cardfile=sherpa_${dataset}_crdE.tgz
libsfile=sherpa_${dataset}_libs.tgz
crssfile=sherpa_${dataset}_crss.tgz
gridfile=sherpa_${dataset}_migr.tgz
if [ ! "${cfdc}" = "" ]; then cardfile=${cfdc}; fi
if [ ! "${cflb}" = "" ]; then libsfile=${cflb}; fi
if [ ! "${cfcr}" = "" ]; then crssfile=${cfcr}; fi
if [ ! "${cfgr}" = "" ]; then gridfile=${cfgr}; fi



if [ ! "${imode}" = "CRAB" ]; then

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
  file_copy ${datadir} ${libsfile} ${PWD}
  file_copy ${datadir} ${crssfile} ${PWD}
  file_copy ${datadir} ${gridfile} ${PWD}
  if [ -e ${cardfile} ]; then
    tar -xzf ${cardfile} && rm ${cardfile}
  else
    echo " <E> file not found: "${cardfile}
    exit 1
  fi
  if [ -e ${libsfile} ]; then
    tar -xzf ${libsfile} && rm ${libsfile}
  else
    echo " <E> file not found: "${libsfile}
    exit 1
  fi
  if [ -e ${crssfile} ]; then
    tar -xzf ${crssfile} && rm ${crssfile}
  else
    echo " <E> file not found: "${crssfile}
    exit 1
  fi
  if [ -e ${gridfile} ]; then
    tar -xzf ${gridfile} && rm ${gridfile}
  else
    echo " <W> no MI grid file: "${gridfile}
  fi
  cd -

fi


# generate & compile pyhton script
if [ "${imode}" = "LOCAL" ]; then
#  cd ${MYCMSSWTEST}
  cd ${MYCMSSWSHPA}
  shpacfifile="sherpa_cfi.py"
  build_python_cfi ${shpacfifile}
 rm *.dat
  mv ${shpacfifile}   ${MYCMSSWPYTH}
#  mv sherpa_custom_cff.py ${MYCMSSWPYTH}
  cd ${MYCMSSWTEST}
  shpacfgfile="sherpa_cfg.py"
  shpaoutfile="sherpa_out.root"
  build_python_cfg ${shpacfgfile} ${shpaoutfile} ${shpacfifile} ${MYANADIR}
  cd ..
  scramv1 b
  cd -
  cd ${HDIR}
fi

if [ "${imode}" = "PROD" ]; then
  shpamstfile="sherpa_"${dataset}"_MASTER.tgz"
  shpamstmd5s="sherpa_"${dataset}"_MASTER.md5"
  shpacfffile="sherpa_"${dataset}"_MASTER_cff.py"

  cd ${MYCMSSWSHPA}

  tar -czf ${shpamstfile} *
####
  md5sum ${shpamstfile} > ${shpamstmd5s}
  summd5=`md5sum   ${shpamstfile} | cut -f1 -d" "` 
#  echo ${summd5}"  "${shpamstfile} > ${shpamstmd5s}
####

  build_python_cfi ${shpacfffile} ${dataset} ${summd5}

  mv ${shpamstfile} $HDIR
  mv ${shpamstmd5s} $HDIR
  mv ${shpacfffile} $HDIR

  cd $HDIR

  rm -rf $TDIR


fi

