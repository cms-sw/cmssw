#!/bin/bash
#
#  file:        PrepareSherpaLibs.sh
#  description: prepare SHERPA libraries and cross sections for local use
#               generate CMSSW python script
#  uses:        SHERPA datacards, libs and cross sections
#
#  author:      Markus Merschmeyer, RWTH Aachen
#  date:        8th July 2011
#  version:     3.1
#  changed: 	Martin Niegel, KIT, 2011/06/07
#		Fix for Sherpa 1.3.0


# +-----------------------------------------------------------------------------------------------+
# function definitions
# +-----------------------------------------------------------------------------------------------+

function print_help() {
    echo "" && \
    echo "PrepareSherpaLibs version 3.1" && echo && \
    echo "options: -i  path       path to SHERPA datacard, library & cross section files" && \
    echo "                         can also be in WWW (http://...) or SE (srm://...)" && \
    echo "                         -> ( "${datadir}" )" && \
    echo "         -p  process    SHERPA dataset/process name ( "${dataset}" )" && \
    echo "         -m  mode       CMSSW running mode ( "${imode}" )" && \
    echo "                         [ 'LOCAL'  : local running of CMSSW         ]" && \
    echo "                         [ 'CRAB'   : prepare crab files in addition ]" && \
    echo "                         [ 'PROD'   : for production validation      ]" && \
    echo "         -c  condition  running conditions ( "${MYCONDITIONS}" )" && \
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

  echo "import FWCore.ParameterSet.Config as cms"                          >> ${shpacfifile}
  echo "import os"                                                         >> ${shpacfifile} 
  echo ""                                                                  >> ${shpacfifile}
  echo "source = cms.Source(\"EmptySource\")"                              >> ${shpacfifile}
  echo ""                                                                  >> ${shpacfifile}
  echo "generator = cms.EDFilter(\"SherpaGeneratorFilter\","               >> ${shpacfifile}
  echo "  maxEventsToPrint = cms.untracked.int32(0),"                      >> ${shpacfifile}
  echo "  filterEfficiency = cms.untracked.double(1.0),"                   >> ${shpacfifile}
  echo "  crossSection = cms.untracked.double(-1),"                        >> ${shpacfifile}
  echo "  Path = cms.untracked.string(os.getcwd()+'/"${MYLIBDIR}"'),"      >> ${shpacfifile}
  echo "  PathPiece = cms.untracked.string(os.getcwd()+'/"${MYLIBDIR}"')," >> ${shpacfifile}
  echo "  ResultDir = cms.untracked.string('Result'),"                     >> ${shpacfifile}
  echo "  default_weight = cms.untracked.double(1.0),"                     >> ${shpacfifile}
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


# function to generate CRAB configuration file
function build_crab_cfg() {

  crabcfgfile=$1 # e.g. "crab.cfg"
  crabpset=$2    # e.g. "XXX_cfg.py"
  crabnevt=$3    # e.g. 10
  craboutf=$4    # e.g. "XXX_GEN.root"
  crabshfile=$5  # e.g. "crab_mm.sh"
  crabpst2=$6    # second parameter set (e.g. for RECO)

#  if [ "${crabsrmpth}" = "./" ]; then # adjust copy data flag
#    iretdata=1
#    icpydata=0
#  else
    iretdata=0
    icpydata=1
#  fi

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
  echo "#events_per_job = 1000"                >> ${crabcfgfile}
  echo "number_of_jobs = 1"                   >> ${crabcfgfile}
  echo "output_file = "${craboutf}            >> ${crabcfgfile}
  echo ""                                     >> ${crabcfgfile}
  echo "[USER]"                               >> ${crabcfgfile}
  echo "script_exe = "${crabshfile}           >> ${crabcfgfile}
  echo "return_data = "${iretdata}            >> ${crabcfgfile}
  echo "copy_data = "${icpydata}              >> ${crabcfgfile}
  echo "storage_element = grid-srm.physik.rwth-aachen.de" >> ${crabcfgfile}
  echo "storage_path = /pnfs/physik.rwth-aachen.de/dcms/merschm" >> ${crabcfgfile}
  echo "user_remote_dir = RES" >> ${crabcfgfile}
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

  echo "#!/bin/bash"                                                                     >> ${crabshfile}
  echo "HDIR=\$PWD"                                                                      >> ${crabshfile}
  echo "SHERPATWIKI="${SHERPATWIKI}                                                      >> ${crabshfile}
  echo "PROCESS_LOC="${PROCESS_LOC}                                                      >> ${crabshfile}
  echo "PROCESS_ID="${PROCESS_ID}                                                        >> ${crabshfile}
  echo "SUBDIR="${SUBDIR}                                                                >> ${crabshfile}
  echo ""                                                                                >> ${crabshfile}
  echo "wget \${SHERPATWIKI}/PrepareSherpaLibs.sh"                                       >> ${crabshfile}
  echo "chmod u+x PrepareSherpaLibs.sh"                                                  >> ${crabshfile}
  echo "./PrepareSherpaLibs.sh -i \${PROCESS_LOC} -p \${PROCESS_ID} -m GRID"             >> ${crabshfile}
  echo ""                                                                                >> ${crabshfile}
#
#  echo " echo \">>>> PWD: \";   pwd"                                                     >> ${crabshfile}
#  echo " echo \">>>> ls -l: \"; ls -l"                                                   >> ${crabshfile}
#  echo " find ./ -name Run.dat"                                                          >> ${crabshfile}
#
  echo "eval \`scramv1 ru -sh\`"                                                         >> ${crabshfile}
  echo "cmsRun -p pset.py"                                                               >> ${crabshfile}
  if [ ! "${crabpst2}" = "" ]; then
  echo ""                                                                                >> ${crabshfile}
  echo "cmsRun -p "${crabpst2}                                                           >> ${crabshfile}
  fi
  echo ""                                                                                >> ${crabshfile}
  echo "cd \$CMSSW_BASE"                                                                 >> ${crabshfile}
  echo "TIME=\`date +%y%m%d_%H%M%S_%N\`"                                                 >> ${crabshfile}
  echo "for FILEIN in \`ls *.root\`; do"                                                 >> ${crabshfile}
  echo "  cnt=0"                                                                         >> ${crabshfile}
  echo "  TEST=\$FILEIN"                                                                 >> ${crabshfile}
  echo "  while [ ! \"\$TEST\" = \"\" ]; do"                                             >> ${crabshfile}
  echo "    let cnt=\$cnt+1"                                                             >> ${crabshfile}
  echo "    TEST=\`echo \$FILEIN | cut -f \$cnt-99 -d\"_\"\`"                            >> ${crabshfile}
  echo "  done"                                                                          >> ${crabshfile}
  echo "  let cnt=\$cnt-1"                                                               >> ${crabshfile}
  echo "  TEST=\`echo \$FILEIN | cut -f \$cnt -d\"_\"\`"                                 >> ${crabshfile}
  echo "  TST1=\`echo \$TEST | cut -f1 -d\".\"\`"                                        >> ${crabshfile}
  echo "  TST2=\`echo \$TEST | cut -f2 -d\".\"\`"                                        >> ${crabshfile}
  echo "  FILEOUT=\"sherpa_\"\$PROCESS_ID\"_\"\$TST1\"_\"\$TIME\".\"\$TST2"              >> ${crabshfile}
  echo "  srmcp file:///\$FILEIN "${crabsrmpth}"/\$FILEOUT"                              >> ${crabshfile}
  echo "  srm-set-permissions -type=ADD -other=W -group=W "${crabsrmpth}"/\$FILEOUT"     >> ${crabshfile}
  echo "  rm \$FILEIN"                                                                   >> ${crabshfile}
  echo "done"                                                                            >> ${crabshfile}
  echo "cd -"                                                                            >> ${crabshfile}
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
imode="LOCAL"                                        # CMSSW running mode
MYANADIR="A/B"                                       # user analysis directory inside CMSSW
#                                                    # -> CMSSW_X_Y_Z/src/${MYANADIR}/
cfdc=""                                              # custom data card file name
cflb=""                                              # custom library file name
cfcr=""                                              # custom cross section file name
MYSRMPATH="./"                                       # SRM path for storage of results
MYLIBDIR="SherpaRun"                                 # name of directory for process-dep. Sherpa files
#MYCONDITIONS="IDEAL_V11"                             # CMSSW_2_2_X conditions
MYCONDITIONS="MC_31X_V5"                             # CMSSW_3_1/2_X conditions
#MYCONDITIONS="STARTUP31X_V4"                         # CMSSW_3_1/2_X conditions
#MYCONDITIONS="DESIGN_31X_V4"                         # CMSSW_3_1/2_X conditions

# get & evaluate options
while getopts :i:p:d:m:c:a:D:L:C:P:h OPT
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
  MYCMSSWTEST=${HDIR}
  MYCMSSWPYTH=${HDIR}
  MYCMSSWSHPA=${HDIR}/${MYLIBDIR}
else
  MYCMSSWTEST=${CMSSWDIR}/src/${MYANADIR}/test
  MYCMSSWPYTH=${CMSSWDIR}/src/${MYANADIR}/python
  MYCMSSWSHPA=${CMSSWDIR}/src/${MYANADIR}/test/${MYLIBDIR}
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
if [ ! "${cfdc}" = "" ]; then cardfile=${cfdc}; fi
if [ ! "${cflb}" = "" ]; then libsfile=${cflb}; fi
if [ ! "${cfcr}" = "" ]; then crssfile=${cfcr}; fi



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
  cd -

fi


# generate & compile pyhton script
if [ "${imode}" = "LOCAL" ] || [ "${imode}" = "CRAB" ]; then
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
  cd ${MYCMSSWSHPA}
  shpacfffile="sherpa_"${dataset}"_cff.py"
  build_python_cfi ${shpacfffile}
 rm *.dat
  mv ${shpacfffile}   ${HDIR}
#  mv sherpa_custom_cff.py ${HDIR}
  cd ${HDIR}
#  tar -czf sherpa_${dataset}_MASTER.tgz ${shpacfffile} sherpa_custom_cff.py ${MYLIBDIR}
  tar -czf sherpa_${dataset}_MASTER.tgz ${shpacfffile} ${MYLIBDIR}
#
#  rm -rf ${shpacfffile} sherpa_custom_cff.py
  rm -rf ${shpacfffile}
#
  rm -rf ${MYLIBDIR}
fi



if [ "${imode}" = "CRAB" ]; then

#  nevts=1000
  nevts=10

  cd ${MYCMSSWTEST}

  crabcfgfile="crab_GEN.cfg"
  crabshfile="crab_GEN.sh"
  SHERPATWIKI="https://twiki.cern.ch/twiki/pub/CMS/SherpaInterface"
  build_crab_cfg ${crabcfgfile} ${shpacfgfile} ${nevts} ${shpaoutfile} ${crabshfile}
  build_crab_sh ${crabshfile} ${SHERPATWIKI} ${datadir} ${dataset} ${MYANADIR} ${MYSRMPATH}


### NEW FEATURE, BE CAREFUL
# build RAW analysis script
  cmsdrvpyfile1="sherpa_cmsdrv_RAW.py"
  cmsdrvanaseq1="GEN,SIM,DIGI,L1,DIGI2RAW,HLT"
  cmsdrvevtcnt1="RAWSIM"
  cmsdrvoutfil1="sherpa_cmsdrv_RAW.root"
  CMD="cmsDriver.py ${MYANADIR}/python/${shpacfifile} -s ${cmsdrvanaseq1} \
                --eventcontent ${cmsdrvevtcnt1} --fileout ${cmsdrvoutfil1} \
                --conditions FrontierConditions_GlobalTag,${MYCONDITIONS}::All \
                -n ${nevts} --python_filename ${cmsdrvpyfile1} --no_exec"
#                -n ${nevts} --python_filename ${cmsdrvpyfile1} --no_exec \
#                --customise ${MYANADIR}/sherpa_custom_cff.py"
  echo "command: "${CMD}
  ${CMD}

# build RECO analysis script
  cmsdrvpyfile2="sherpa_cmsdrv_RECO.py"
  cmsdrvanaseq2="RAW2DIGI,RECO"
  cmsdrvevtcnt2="RECOSIM"
  cmsdrvoutfil2="sherpa_cmsdrv_RECO.root"
  CMD="cmsDriver.py reco -s ${cmsdrvanaseq2} --filein file:${cmsdrvoutfil1} \
                --eventcontent ${cmsdrvevtcnt2} --fileout ${cmsdrvoutfil2} \
                --conditions FrontierConditions_GlobalTag,${MYCONDITIONS}::All \
                -n -1 --python_filename ${cmsdrvpyfile2} --no_exec"
  echo "command: "${CMD}
  ${CMD}

# build crab scripts
  crabcfgfile="crab_cmsdrv.cfg"
  crabshfile="crab_cmsdrv.sh"
  SHERPATWIKI="https://twiki.cern.ch/twiki/pub/CMS/SherpaInterface"
  build_crab_cfg ${crabcfgfile} ${cmsdrvpyfile1} ${nevts} ${cmsdrvoutfil2} ${crabshfile} ${cmsdrvpyfile2}
  build_crab_sh ${crabshfile} ${SHERPATWIKI} ${datadir} ${dataset} ${MYANADIR} ${MYSRMPATH} ${cmsdrvpyfile2}
### NEW FEATURE, BE CAREFUL

  cd ${HDIR}

fi
