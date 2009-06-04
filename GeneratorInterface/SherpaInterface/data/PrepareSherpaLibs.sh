#!/bin/bash
#
#  file:        PrepareSherpaLibs.sh
#  description: prepare SHERPA libraries and cross sections for local use
#               generate CMSSW python script
#  uses:        SHERPA datacards, libs and cross sections
#
#  author:      Markus Merschmeyer, RWTH Aachen
#  date:        2008/11/28
#  version:     2.5
#



# +-----------------------------------------------------------------------------------------------+
# function definitions
# +-----------------------------------------------------------------------------------------------+

function print_help() {
    echo "" && \
    echo "PrepareSherpaLibs version 2.5" && echo && \
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
    echo "         -P  SRM path   (CRAB) SE path for final results" && \
    echo "                         -> ( "${MYSRMPATH}" )" && \
    echo "         -h             display this help and exit" && echo
}


# function to build a python script for cmsDriver
function build_python_cfi() {

  shpacfifile=$1  # config file name

  if [ -e ${shpacfifile} ]; then rm ${shpacfifile}; fi
  touch ${shpacfifile}

cat >> ${shpacfifile} << EOF
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

  shpacfgfile=$1  # include file name
  shpaoutfile=$2  # output root file name

  if [ -e ${shpacfgfile} ]; then rm ${shpacfgfile}; fi
  touch ${shpacfgfile}

cat >> ${shpacfgfile} << EOF
import FWCore.ParameterSet.Config as cms

process = cms.Process("runSherpa")
process.source=cms.Source("SherpaSource",
  firstRun  = cms.untracked.uint32(1),
  libDir    = cms.untracked.string('SherpaRun'),
  resultDir = cms.untracked.string('Result')
)
EOF
  echo "process.RandomNumberGeneratorService = cms.Service(\"RandomNumberGeneratorService\"," >> ${shpacfgfile}
  echo "    sourceSeed = cms.untracked.uint32(98765)"                                         >> ${shpacfgfile}
  echo ")"                                                                                    >> ${shpacfgfile}
  echo "process.maxEvents = cms.untracked.PSet("                                              >> ${shpacfgfile}
  echo "    input = cms.untracked.int32(100)"                                                 >> ${shpacfgfile}
  echo ")"                                                                                    >> ${shpacfgfile}
  echo "process.sherpa_out = cms.OutputModule(\"PoolOutputModule\","                          >> ${shpacfgfile}
  echo "    fileName = cms.untracked.string('"${shpaoutfile}"')"                              >> ${shpacfgfile}
  echo ")"                                                                                    >> ${shpacfgfile}
  echo "process.outpath = cms.EndPath(process.sherpa_out)"                                    >> ${shpacfgfile}
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


# function to generate CRAB configuration file
function build_crab_cfg() {

  crabcfgfile=$1 # e.g. "crab.cfg"
  crabpset=$2    # e.g. "XXX_cfg.py"
  crabnevt=$3    # e.g. 10
  craboutf=$4    # e.g. "XXX_GEN.root"
  crabshfile=$5  # e.g. "crab_mm.sh"
  crabsrmpth=$6  # e.g. "srm://grid-srm.physik.rwth-aachen.de:8443//srm/managerv1\?SFN=/pnfs/..."
#!!!             #      "srm://grid-srm.physik.rwth-aachen.de:8443//pnfs/physik.rwth-aachen.de/dcms/merschm/RES"
  crabpst2=$7    # second parameter set (e.g. for RECO)

  if [ "${crabsrmpth}" = "./" ]; then # adjust copy data flag
    iretdata=1
    icpydata=0
  else
    iretdata=0
    icpydata=1
  fi
### quick fix
#  iretdata=0
#  icpydata=0
###

# disentangle storage path
  CRABSE=`echo ${crabsrmpth} | cut -f2 -d":" | cut -f3 -d"/"`
  CRABTMP=`echo ${crabsrmpth} | cut -f3 -d":" | cut -f2-99 -d"/"`
  CRABLF1=`echo ${CRABTMP} | cut -f1 -d"="`
  CRABLF2=`echo ${CRABTMP} | cut -f2 -d"="`
  cnt=0
  while [ ! "${CRABLF2}" = "" ]; do
    let cnt=$cnt+1
    CRABLF2=`echo ${CRABLF2} | cut -f $cnt-99 -d"."`
  done
  CRABLF3=`echo ${CRABTMP} | cut -f2 -d"=" | cut -f $cnt -d"." | cut -f1 -d"/"`
  let cnt=$cnt-1
  CRABLF2=`echo ${CRABTMP} | cut -f2 -d"=" | cut -f 1-$cnt -d"."`
  if [ `echo ${CRABTMP} | grep -c "="` -gt 0 ]; then
#    CRABSP=${CRABLF1}"="${CRABLF2}"."${CRABLF3}"/"
    CRABSP="/"${CRABLF1}"="${CRABLF2}"."${CRABLF3}
  else
#    CRABSP=${CRABLF2}"."${CRABLF3}"/"
    CRABSP="/"${CRABLF2}"."${CRABLF3}
  fi
  let cnt=$cnt+1
  CRABLFN=`echo ${CRABTMP} | cut -f2 -d"=" | cut -f $cnt -d"." | cut -f2-99 -d"/"`
#  CRABLFN="/"${CRABLFN}
  CRABLFN=${CRABLFN}

  if [ -e ${crabcfgfile} ]; then rm ${crabcfgfile}; fi
  touch ${crabcfgfile}

cat >> ${crabcfgfile} << EOF
[CRAB]
jobtype = cmssw
scheduler = glite

[CMSSW]
datasetpath=none
EOF
  echo "pset = "${crabpset}                   >> ${crabcfgfile}
  echo "total_number_of_events = "${crabnevt} >> ${crabcfgfile}
#  echo "events_per_job = 1000"                >> ${crabcfgfile}
  echo "events_per_job = 10"                >> ${crabcfgfile}
  echo "#number_of_jobs = 5"                  >> ${crabcfgfile}
  echo "output_file = "${craboutf}            >> ${crabcfgfile}
  echo ""                                     >> ${crabcfgfile}
  echo "[USER]"                               >> ${crabcfgfile}
  echo "script_exe = "${crabshfile}           >> ${crabcfgfile}
#  echo "return_data = 1"                      >> ${crabcfgfile}
  echo "return_data = "${iretdata}            >> ${crabcfgfile}
#  echo "copy_data = 0"                        >> ${crabcfgfile}
  echo "copy_data = "${icpydata}              >> ${crabcfgfile}
  echo "storage_element = "${CRABSE}          >> ${crabcfgfile}
  echo "storage_path = "${CRABSP}             >> ${crabcfgfile}
  echo "lfn = "${CRABLFN}                     >> ${crabcfgfile}
if [ ! "${crabpst2}" = "" ]; then
  echo "additional_input_files = "${crabpst2} >> ${crabcfgfile}
fi
  echo ""                                     >> ${crabcfgfile}
cat >> ${crabcfgfile} << EOF

[EDG]
rb = CERN
se_black_list = T0,T1
#se_white_list =
#ce_black_list =
ce_white_list = rwth-aachen.de,cern.ch,infn.it,fnal.gov

[CONDORG]
#batchsystem = condor

EOF
}


# function to generate CRAB executable file
function build_crab_sh() {
  crabshfile=$1  # e.g. "crab_mm.sh"
  SHERPATWIKI=$2 # e.g. "https://twiki.cern.ch/twiki/pub/CMS/SherpaInterface"
  PROCESS_LOC=$3 # e.g. "https://twiki.cern.ch/twiki/pub/CMS/SherpaInterface"
  PROCESS_ID=$4  # e.g. "TEST"
  SUBDIR=$5      # e.g. "TEST1/TEST2"
  crabsrmpth=$6  # e.g. "srm://grid-srm.physik.rwth-aachen.de:8443//srm/managerv1\?SFN=/pnfs/..."
#!!!             #      "srm://grid-srm.physik.rwth-aachen.de:8443//pnfs/physik.rwth-aachen.de/dcms/merschm/RES"
  crabpst2=$7    # second parameter set (e.g. for RECO)

  if [ -e ${crabshfile} ]; then rm ${crabshfile}; fi
  touch ${crabshfile}; chmod u+x ${crabshfile}

  echo "#!/bin/bash"                                                                               >> ${crabshfile}
  echo "# setup"                                                                                   >> ${crabshfile}
  echo "HDIR=\$PWD"                                                                                >> ${crabshfile}
  echo "SHERPATWIKI="${SHERPATWIKI}                                                                >> ${crabshfile}
  echo "PROCESS_LOC="${PROCESS_LOC}                                                                >> ${crabshfile}
  echo "PROCESS_ID="${PROCESS_ID}                                                                  >> ${crabshfile}
  echo "SUBDIR="${SUBDIR}                                                                          >> ${crabshfile}
  echo ""                                                                                          >> ${crabshfile}
  echo "# setup (fix) CMSSW + SHERPA"                                                              >> ${crabshfile}
  echo "export MYSHERPAPATH=\`scramv1 tool info sherpa | grep SHERPA_BASE | cut -f2 -d\"=\"\`"     >> ${crabshfile}
  echo "export SHERPA_SHARE_PATH=\$MYSHERPAPATH/share/SHERPA-MC"                                   >> ${crabshfile}
  echo "export SHERPA_INCLUDE_PATH=\$MYSHERPAPATH/include/SHERPA-MC"                               >> ${crabshfile}
  echo ""                                                                                          >> ${crabshfile}
  echo "# setup SHERPA library and cross sections"                                                 >> ${crabshfile}
  echo "wget \${SHERPATWIKI}/PrepareSherpaLibs.sh"                                                 >> ${crabshfile}
  echo "chmod u+x PrepareSherpaLibs.sh"                                                            >> ${crabshfile}
  echo "./PrepareSherpaLibs.sh -i \${PROCESS_LOC} -p \${PROCESS_ID} -d \$CMSSW_BASE -a \${SUBDIR}" >> ${crabshfile}
  echo "mv \$CMSSW_BASE/src/\${SUBDIR}/test/SherpaRun ."                                           >> ${crabshfile}
  echo ""                                                                                          >> ${crabshfile}
  echo "# run CMSSW"                                                                               >> ${crabshfile}
  echo "eval \`scramv1 ru -sh\`"                                                                   >> ${crabshfile}
  echo "cmsRun -p pset.py"                                                                         >> ${crabshfile}
## 
## copy files manually & fix file permissions here
## 
  echo "pwd"                                                                                       >> ${crabshfile}
  echo "ls -l"                                                                                     >> ${crabshfile}
  echo ""                                                                                          >> ${crabshfile}
  echo "cmsRun -p "${crabpst2}                                                                     >> ${crabshfile}
  echo "pwd"                                                                                       >> ${crabshfile}
  echo "ls -l"                                                                                     >> ${crabshfile}
  echo ""                                                                                          >> ${crabshfile}
  echo ""                                                                                          >> ${crabshfile}
  echo "cd \$CMSSW_BASE"                                                                           >> ${crabshfile}
  echo "TIME=\`date +%y%m%d_%H%M%S_%N\`"                                                           >> ${crabshfile}
  echo "for FILEIN in \`ls *.root\`; do"                                                           >> ${crabshfile}
  echo "  cnt=0"                                                                                   >> ${crabshfile}
  echo "  TEST=\$FILEIN"                                                                           >> ${crabshfile}
  echo "  while [ ! \"\$TEST\" = \"\" ]; do"                                                       >> ${crabshfile}
  echo "    let cnt=\$cnt+1"                                                                       >> ${crabshfile}
  echo "    TEST=\`echo \$FILEIN | cut -f \$cnt-99 -d\"_\"\`"                                      >> ${crabshfile}
  echo "  done"                                                                                    >> ${crabshfile}
  echo "  let cnt=\$cnt-1"                                                                         >> ${crabshfile}
  echo "  TEST=\`echo \$FILEIN | cut -f \$cnt -d\"_\"\`"                                           >> ${crabshfile}
  echo "  TST1=\`echo \$TEST | cut -f1 -d\".\"\`"                                                  >> ${crabshfile}
  echo "  TST2=\`echo \$TEST | cut -f2 -d\".\"\`"                                                  >> ${crabshfile}
  echo "  FILEOUT=\"sherpa_\"\$PROCESS_ID\"_\"\$TST1\"_\"\$TIME\".\"\$TST2"                        >> ${crabshfile}
  echo "  srmcp file:///\$FILEIN "${crabsrmpth}"/\$FILEOUT"                                        >> ${crabshfile}
  echo "  srm-set-permissions -type=ADD -other=W -group=W "${crabsrmpth}"/\$FILEOUT"               >> ${crabshfile}
  echo "  rm \$FILEIN"                                                                             >> ${crabshfile}
  echo "done"                                                                                      >> ${crabshfile}
  echo "cd -"                                                                                      >> ${crabshfile}
##
##
##
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
MYSRMPATH="./"                                         # SRM path for storage of results
#MYSRMPATH="srm://grid-srm.physik.rwth-aachen.de:8443//srm/managerv1\?SFN=/pnfs/physik.rwth-aachen.de/dcms/merschm"

# get & evaluate options
while getopts :i:p:o:d:m:a:D:L:C:P:h OPT
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
  rm -f ${CMSSWDIR}/python/${MYANADIR}/*.py*                  # ...clean up
fi
if [ ! -e ${CMSSWDIR}/src/${MYANADIR}/python ]; then          # create 'python' subdirectory
  mkdir -p ${CMSSWDIR}/src/${MYANADIR}/python
else
  rm -f ${CMSSWDIR}/src/${MYANADIR}/python/*.py*              # ...clean up
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
  if [ -e ${cardfile} ]; then
    tar -xzf ${cardfile} && rm ${cardfile}
  else
    exit 0
  fi
  file_copy ${datadir} ${libsfile} ${PWD}
  if [ -e ${libsfile} ]; then
    tar -xzf ${libsfile} && rm ${libsfile}
  else
    exit 0
  fi
  if [ "${dataopt}" = "LBCR" ]; then
    file_copy ${datadir} ${crssfile} ${PWD}
    if [ -e ${crssfile} ]; then
      tar -xzf ${crssfile} && rm ${crssfile}
    else
      exit 0
    fi
  fi
  cd -

fi

# generate & compile pyhton script

cd ${MYCMSSWTEST}
shpacfifile="sherpa_cfi.py"
shpacfgfile="sherpa_cfg.py"
shpaoutfile="sherpa_out.root"
build_python_cfi ${shpacfifile}
mv ${shpacfifile} ../python/
build_python_cfg ${shpacfgfile} ${shpaoutfile}
cd ..
scramv1 b
cd -
cd ${HDIR}

if [ "${imode}" = "CRAB" ]; then

#  nevts=1000
  nevts=30

  cd ${MYCMSSWTEST}

  crabcfgfile="crab_GEN.cfg"
  crabshfile="crab_GEN.sh"
  SHERPATWIKI="https://twiki.cern.ch/twiki/pub/CMS/SherpaInterface"
  build_crab_cfg ${crabcfgfile} ${shpacfgfile} ${nevts} ${shpaoutfile} ${crabshfile} ${MYSRMPATH}
  build_crab_sh ${crabshfile} ${SHERPATWIKI} ${datadir} ${dataset} ${MYANADIR}


### NEW FEATURE, BE CAREFUL
# build RAW analysis script
  cmsdrvpyfile1="sherpa_cmsdrv_RAW.py"
  cmsdrvanaseq1="GEN,SIM,DIGI,L1,DIGI2RAW,HLT"
  cmsdrvevtcnt1="RAWSIM"
  cmsdrvoutfil1="sherpa_cmsdrv_RAW.root"
  CMD="cmsDriver.py ${MYANADIR}/python/${shpacfifile} -s ${cmsdrvanaseq1} \
                --eventcontent ${cmsdrvevtcnt1} --fileout ${cmsdrvoutfil1} \
                --conditions FrontierConditions_GlobalTag,IDEAL_V9_900::All \
                -n 10 --magField 3.8T --python_filename ${cmsdrvpyfile1} \
                --no_exec --dump_python"
  echo "command: "${CMD}
  ${CMD}
# fix "abort on unknown PDG code" issue
  sed -e 's/PDGCode = cms.untracked.bool(True)/PDGCode = cms.untracked.bool(False)/' < ${cmsdrvpyfile1} > ${cmsdrvpyfile1}.tmp
  mv ${cmsdrvpyfile1}.tmp ${cmsdrvpyfile1}
# build RECO analysis script
  cmsdrvpyfile2="sherpa_cmsdrv_RECO.py"
  cmsdrvanaseq2="RAW2DIGI,RECO"
  cmsdrvevtcnt2="RECOSIM"
  cmsdrvoutfil2="sherpa_cmsdrv_RECO.root"
  CMD="cmsDriver.py reco -s ${cmsdrvanaseq2} --filein file:${cmsdrvoutfil1} \
                --eventcontent ${cmsdrvevtcnt2} --fileout ${cmsdrvoutfil2} \
                --conditions FrontierConditions_GlobalTag,IDEAL_V9_900::All \
                -n -1 --magField 3.8T --python_filename ${cmsdrvpyfile2} \
                --no_exec"
  echo "command: "${CMD}
  ${CMD}
# build crab scripts
  crabcfgfile="crab_cmsdrv.cfg"
  crabshfile="crab_cmsdrv.sh"
  SHERPATWIKI="https://twiki.cern.ch/twiki/pub/CMS/SherpaInterface"
##  build_crab_cfg ${crabcfgfile} ${cmsdrvpyfile1} ${nevts} ${cmsdrvoutfil1} ${crabshfile} ${MYSRMPATH} ${cmsdrvpyfile2}
  build_crab_cfg ${crabcfgfile} ${cmsdrvpyfile1} ${nevts} ${cmsdrvoutfil2} ${crabshfile} ${MYSRMPATH} ${cmsdrvpyfile2}
  build_crab_sh ${crabshfile} ${SHERPATWIKI} ${datadir} ${dataset} ${MYANADIR} ${MYSRMPATH} ${cmsdrvpyfile2}
### NEW FEATURE, BE CAREFUL

  cd ${HDIR}

fi
