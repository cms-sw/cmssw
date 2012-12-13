#!/bin/bash
#
#  file:        MakeSherpaLibs.sh
#  description: BASH script handling the SHERPA data cards for the
#               library production and cross section calculation
#  uses:        the required SHERPA data cards (+ libraries) [see below]
#
#  author:      Markus Merschmeyer, Sebastian Thueer, RWTH Aachen
#  date:        13th December 2012
#  version:     4.1
#



# +-----------------------------------------------------------------------------------------------+
# function definitions
# +-----------------------------------------------------------------------------------------------+

print_help() {
    echo "" && \
    echo "MakeSherpaLibs version 4.0" && echo && \
    echo "options: -d  path       (optional) path to your SHERPA installation (otherwise the SHERPA" && \
    echo "                         package belonging to the release under '\$CMSSW_BASE' is used)" && \
    echo "                         -> ( "${shr}" )" && \
    echo "         -i  path       path to SHERPA datacard (and library, see -o) files" && \
    echo "                         -> ( "${inc}" )" && \
    echo "         -p  process    SHERPA process/dataset name ( "${prc}" )" && \
    echo "         -o  options    library/cross section options ( "${lbo}" )" && \
    echo "                         [ 'LBCR' : generate libraries and cross sections     ]" && \
    echo "                         [ 'LIBS' : generate libraries only                   ]" && \
    echo "                         [ 'CRSS' : generate cross sections, needs libraries! ]" && \
    echo "                         [ 'EVTS' : generate events, needs libs + crss. sec.! ]" && \
    echo "         -f  path       output path for SHERPA library & cross section files" && \
    echo "                         -> ( "${fin}" )" && \
    echo "         -D  filename   (optional) name of data card file ( "${cfdc}" )" && \
    echo "         -L  filename   (optional) name of library file ( "${cflb}" )" && \
    echo "         -C  filename   (optional) name of cross section file ( "${cfcr}" )" && \
    echo "         -A             switch on multiple interactions in Run.dat card ( "${FLGAMISIC}" )" && \
    echo "         -h             display this help and exit" && echo
}

check_occurence() {
# $1: name of a text file
# $2: string to search in file named $1
# returns: number of occurences of string $2 in file $1
  if [ -e $1 ]; then
    cnt=0
    if [ $# -eq 3 ]; then
      cnt=`cat $1 | grep -i $2 | grep -i -c $3`
    else
      cnt=`cat $1 | grep -i $2 | grep -i -c $2`
    fi
    if [ $cnt -gt 0 ]; then
      echo 1
    else
      echo 0
    fi
  else
    echo " <E> file "$1" not found!"
  fi
}

clean_libs() {
  DIRS=`find Process -name P?_?`" "`find Process -name P?_??`
  for J in $DIRS ; do
    echo "."
    echo "======================"
    echo "$J";
    echo "======================"
    cd $J
    make clean
    rm config* Makefile*
#    rm *.tex
    rm aclocal.m4 ChangeLog depcomp install-sh libtool ltmain.sh missing
    rm AUTHORS COPYING INSTALL NEWS README 
    rm -rf autom4te.cache
    find ./ -type f -name 'Makefile*' -exec rm -rf {} \;
    find ./ -type d -name '.deps'     -exec rm -rf {} \;
    find ./ -type f -name '*.C'       -exec rm -rf {} \;
    find ./ -type f -name '*.H'       -exec rm -rf {} \;
    cd ../..
  done
}




# +-----------------------------------------------------------------------------------------------+
# start of the script
# +-----------------------------------------------------------------------------------------------+

# save current path
HDIR=`pwd`

# dummy setup (if all options are missing)
shr=${HDIR}/SHERPA_1.4.2        # path to SHERPA installation
scrloc=`which scramv1 &> tmp.tmp; cat tmp.tmp | cut -f1 -d"/"; rm tmp.tmp`
if [ "${scrloc}" = "" ]; then
  shr=`scramv1 tool info sherpa | grep "SHERPA_BASE" | cut -f2 -d"="`
  shrinit=$shr
fi
pth="TMP"                          # name of SHERPA data card directory
prc="XXX"                          # SHERPA process name
lbo="LBCR"                         # library/cross section option
inc=${HDIR}                        # path to SHERPA datacards (libraries)
cfdc=""                            # custom data card file name
cflb=""                            # custom library file name
cfcr=""                            # custom cross section file name
fin=${HDIR}                        # output path for SHERPA libraries & cross sections
FLGAMISIC="FALSE"                  # switch on multiple interactions for production
FLGAMEGIC="FALSE"                  # flag to indicate the usage of AMEGIC -> library compilation required

# get & evaluate options
while getopts :d:i:p:o:f:D:L:C:Ah OPT
do
  case $OPT in
  d) shr=$OPTARG ;;
  i) inc=$OPTARG ;;
  p) prc=$OPTARG ;;
  o) lbo=$OPTARG ;;
  f) fin=$OPTARG ;;
  D) cfdc=$OPTARG ;;
  L) cflb=$OPTARG ;;
  C) cfcr=$OPTARG ;;
  A) FLGAMISIC="TRUE" ;;
  h) print_help && exit 0 ;;
  \?)
    shift `expr $OPTIND - 1`
    if [ "$1" = "--help" ]; then print_help && exit 0;
    else 
      echo -n "MakeSherpaLibs: error: unrecognized option "
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
cd ${shr} && shr=`pwd`; cd ${HDIR}
cd ${inc} && inc=`pwd`; cd ${HDIR}
cd ${fin} && fin=`pwd`; cd ${HDIR}

# test whether to take $shr from CMSSW installation
if [ ! "$CMSSW_BASE" = "" ] && [ "$shr" = "$shrinit" ]; then
  newshr=""
  cd $CMSSW_BASE &&
  newshr=`scramv1 tool info sherpa | grep BASE | cut -f 2 -d "="`
  if [ "${newshr}" = "" ]; then
    echo " <E> no 'sherpa' tool defined in CMSSW, are you sure that"
    echo " <E>  1. the command 'scramv1' is available ?"
    echo " <E>  2. the path to your CMSSW is correct ?"
    echo " <E>  3. there exists a SHERPA package in your CMSSW ?"
    exit 0
  fi
  shr=${newshr}
  cd ${HDIR}
fi
if [ "${SHERPA_SHARE_PATH}"   = "" ]; then export SHERPA_SHARE_PATH=${shr}/share/SHERPA-MC;     fi
if [ "${SHERPA_INCLUDE_PATH}" = "" ]; then export SHERPA_INCLUDE_PATH=${shr}/include/SHERPA-MC; fi
if [ "${SHERPA_LIBRARY_PATH}" = "" ]; then export SHERPA_LIBRARY_PATH=${shr}/lib/SHERPA-MC;     fi


# find 'Run' directory
shrun=${HDIR}/SHERPATMP_${prc}
mkdir -p ${shrun}
pth=${shrun}/${pth} #SHERPA 1.3.0 needs full path, MN 070611

echo "  -> SHERPA path: '"${shr}"'"
echo "  -> SHERPA run path: '"${shrun}"'"
echo "  -> PROCESS name: '"${prc}"'"
echo "  -> Library & cross section otions: '"${lbo}"'"
echo "  -> include path: '"${inc}"'"
echo "  -> custom data card file name: '"${cfdc}"'"
echo "  -> custom library file name: '"${cflb}"'"
echo "  -> custom cross section file name: '"${cfcr}"'"
echo "  -> output path: '"${fin}"'"

# get the number of CPU cores
FLGMCORE="TRUE"
POPTS=""
if [ "$FLGMCORE" = "TRUE" ]; then
    nprc=`cat /proc/cpuinfo | grep  -c processor`
    let nprc=$nprc+1
    if [ $nprc -gt 2 ]; then
      echo " <I> multiple CPU cores detected: "$nprc"-1"
      POPTS=" -j"$nprc" "
    fi
fi






### go to 'Run' subdirectory of SHERPA
cd ${shrun}


### set base name for SHERPA output file(s) and directories
runfile="Run.dat"
runcardfile=Run.dat_${prc}
outflbs=sherpa_${prc}
cardfile=${outflbs}_cards.tgz             # input card file (master -> libraries)
if [ "${lbo}" = "CRSS" ]; then
  cardfile=${outflbs}_crdC.tgz
elif [ "${lbo}" = "EVTS" ]; then
  cardfile=${outflbs}_crdE.tgz
fi
libsfile=${outflbs}_libs.tgz              # output libs
crssfile=${outflbs}_crss.tgz              # output cross sections
evtsfile=${outflbs}_evts.tgz              # output events
if [ ! "${cfdc}" = "" ]; then
  cardfile=${cfdc}                        # custom input data card file
  echo " <I> using custom data card file: "${cardfile}
fi
if [ ! "${cflb}" = "" ]; then
  libsfile=${cflb}                        # custom input library file
  echo " <I> using custom library file: "${libsfile}
fi
if [ ! "${cfcr}" = "" ]; then
  crssfile=${cfcr}                        # custom input cross section file
  echo " <I> using custom cross section file: "${crssfile}
fi
crdlfile=${outflbs}_crdL.tgz              # output cardfile (-> from library production)
crdcfile=${outflbs}_crdC.tgz              # output cardfile (-> from cross section calculation)
crdefile=${outflbs}_crdE.tgz              # output cardfile (-> from event generation)
loglfile=${outflbs}_logL.tgz              # output messages (-> from library production)
logcfile=${outflbs}_logC.tgz              # output messages (-> from cross section calculation)
logefile=${outflbs}_logE.tgz              # output messages (-> from event generation)
#
gridfile=${outflbs}_migr.tgz              # multiple interactions phase-space grid
#
dir1="Process"                            # SHERPA process directory name
dir2="Results"                            # SHERPA results directory name
dir3="Analysis"                           # SHERPA analysis directory name

### clean up existing xsection files
for FILE in `ls xsections_*.dat 2> /dev/null`; do
  echo " <W> deleting existing cross section file: "${FILE}
  rm ${FILE}
done


### test whether PATH exists
if [ -e ${pth} ]; then
  echo " path '"${pth}"' exists,"
  echo "  -> cleaning path '"${pth}"'"
  rm -rf ${pth}/*
else
  echo " path '"${pth}"' does not exist,"
  echo "  -> creating path '"${pth}"'"
  mkdir ${pth}
fi
#MM() pth="./"$pth
cd ${pth}
#MM() pth=`pwd`


### get data card (+ library) tarball(s) from include path
if [ -e ${inc}/${cardfile} ]; then
  cp ${inc}/${cardfile} ./
else
  if [ -e ${inc}/${runcardfile} ]; then
    cp ${inc}/${runcardfile} ./
  fi
fi
if [ "${lbo}" = "CRSS" ]; then
  cp ${inc}/${libsfile} ./
fi
if [ "${lbo}" = "EVTS" ]; then
  cp ${inc}/${libsfile} ./
  cp ${inc}/${crssfile} ./
  cp ${inc}/${gridfile} ./
fi


### check existence of data card file
if [ -e ${cardfile} ]; then
  echo " data card file '"${cardfile}"' exists,"
  echo "  -> unpacking data card file"
  tar -xzf ${cardfile}
else
  if [ -e ${runcardfile} ]; then
    mv ${runcardfile} ${runfile}
  else
    echo " <E> no data card file found"
    echo "  -> stopping..."
    exit
  fi
fi


### find out whether COMIX or AMEGIC is being used
if [ -e ${runfile} ]; then
  iamegic=`check_occurence ${runfile} "ME_SIGNAL_GENERATOR" "AMEGIC"`
  if [ ${iamegic} -gt 0 ]; then
    FLGAMEGIC="TRUE"                   # using AMEGIC
    echo " <I> using AMEGIC ME generator"
  else
#    FLGAMEGIC="FALSE"                  # using no AMEGIC
    echo " <I> using COMIX/internal ME generator"
    lbo="LBCR"
  fi
fi
###exit

### reject mixed occurences of Sherpa's "Enhance" options
#runfile="Run.dat"
if [ -e ${runfile} ]; then
  nenhfac=0; nenhfac=`check_occurence ${runfile} "enhance_factor"`
  nenhfnc=0; nenhfnc=`check_occurence ${runfile} "enhance_function"`
  nenhobs=0; nenhobs=`check_occurence ${runfile} "enhance_observable"`
  sumenh=0; let sumenh=$nenhfac+$nenhfnc+$nenhobs
  if [ ${sumenh} -gt 1 ]; then
    echo " <E> mixed occurence of enhance options in "${runfile}
    echo "  -> stopping..."
    exit
  fi
  flgwgt=0; flgwgt=`check_occurence ${runfile} "EVENT_GENERATION_MODE" "Weighted"`
  if [ ${flgwgt} -eq 0 ] && [ ${nenhfnc} -eq 1 ]; then
    echo " <E> unweighted production and enhance_function not supported by Sherpa!"
    echo "  -> stopping..."
    exit
  fi
  if [ ${flgwgt} -eq 0 ] && [ ${nenhobs} -eq 1 ]; then
    echo " <E> unweighted production and enhance_observable currently not supported!"
    echo "  -> stopping..."
    exit
  fi
fi
###exit





### check required subdirectories
## generate/clean 'Process' subdirectory
if [ ! -e ${dir1} ]; then
  echo " '"${dir1}"' subdirectory does not exist and will be created"
  mkdir ${dir1}
else
  echo " cleaning '"${dir1}"' subdirectory"
  rm -rf ${dir1}/*
fi
## generate/clean 'Results' subdirectory
if [ ! -e ${dir2} ]; then
  echo " '"${dir2}"' subdirectory does not exist and will be created"
  mkdir ${dir2}
else
  echo " cleaning '"${dir2}"' subdirectory"
  rm -rf ${dir2}/*
fi

### check, whether only cross sections have to be calculated
if [ "${lbo}" = "CRSS" ] || [ "${lbo}" = "EVTS" ]; then
  if [ -e ${libsfile} ]; then
    echo " <I> library file '"${libsfile}"' exists,"
    echo "  -> unpacking library file"
    tar -xzf ${libsfile}
  else
    echo " <E> no library file found"
    echo "  -> stopping..."
    exit
  fi
fi

### check, whether only events have to be generated
if [ "${lbo}" = "EVTS" ]; then
  if [ -e ${crssfile} ]; then
    echo " <I> cross section file '"${crssfile}"' exists,"
    echo "  -> unpacking cross section file"
    tar -xzf ${crssfile}
  else
    echo " <E> no cross section file found"
    echo "  -> stopping..."
    exit
  fi
  if [ -e ${gridfile} ]; then
    echo " <I> MI grid file '"${gridfile}"' exists,"
    echo "  -> unpacking phase-space grid"
    tar -xzf ${gridfile}
  else
    echo " <W> no MI grid file found"
    echo "  -> will be recreated..."
  fi
fi



### generate process-specific libraries -> redirect output (stdout, stderr) to files
sherpaexe=`find ${shr} -type f -name Sherpa`
echo " <I> Sherpa executable is "${sherpaexe}
cd ${pth}

# create logfiles
touch ${shrun}/${outflbs}_passLC.out
touch ${shrun}/${outflbs}_passLC.err
if [ "${FLGAMEGIC}" == "TRUE" ]; then
  touch ${shrun}/${outflbs}_mklib.out
  touch ${shrun}/${outflbs}_mklib.err
  touch ${shrun}/${outflbs}_cllib.out
  touch ${shrun}/${outflbs}_cllib.err
fi

## first pass (loop if AMEGIC + NLO loop generators are used)
if [ "${lbo}" = "LIBS" ] || [ "${lbo}" = "LBCR" ]; then
  echo " <I> creating library code..."
  ${sherpaexe} "PATH="${pth} "PATH_PIECE="${pth} "RESULT_DIRECTORY="${dir2} 1>>${shrun}/${outflbs}_passLC.out 2>>${shrun}/${outflbs}_passLC.err
###  cd ${pth}

  if [ "${FLGAMEGIC}" == "TRUE" ]; then

    FLGNEWCODE="TRUE"
    FLGWRITLIB="TRUE"

    while [ "${FLGNEWCODE}" = "TRUE" ] || [ "${FLGWRITLIB}" = "TRUE" ]; do

# compile created library code
      echo " <I> compiling libraries..."
      ./makelibs ${POPTS} 1>>${shrun}/${outflbs}_mklib.out 2>>${shrun}/${outflbs}_mklib.err
# get gross size of created libraries
      nf=`du -sh | grep -o "\." | grep -c "\."`
      lsize=`du -sh  | cut -f 1-${nf} -d "."`
      echo " <I>  -> raw size: "${lsize}
      echo " <I> cleaning libraries..."
      clean_libs 1>>${shrun}/${outflbs}_cllib.out 2>>${shrun}/${outflbs}_cllib.err
# get net size of created libraries
      nf=`du -sh | grep -o "\." | grep -c "\."`
      lsize=`du -sh  | cut -f 1-${nf} -d "."`
      echo " <I>  -> clean size: "${lsize}

# reinvoke Sherpa
      echo " <I> re-invoking Sherpa for futher library/cross section calculation"
      ${sherpaexe} "PATH="${pth} "PATH_PIECE="${pth} "RESULT_DIRECTORY="${dir2} 1>>${shrun}/${outflbs}_passLC.out 2>>${shrun}/${outflbs}_passLC.err

# newly created process code by AMEGIC?
      cd ${dir1}
      lastdir=`ls -C1 -drt * | tail -1`
      npdir=`echo ${lastdir} | grep -c "P2"`
      if [ ${npdir} -gt 0 ]; then
        echo " <I> (AMEGIC) library code was created in (at least) "${lastdir}
        FLGNEWCODE="TRUE"
      else
        FLGNEWCODE="FALSE"
      fi
      cd ${pth}

# mentioning of "" in last 100 lines output file?
      nlines=200
      nphbw=`tail -${nlines} ${shrun}/${outflbs}_passLC.out | grep -c "has been written"`
      npasw=`tail -${nlines} ${shrun}/${outflbs}_passLC.out | grep -c "AMEGIC::Single_Process::WriteLibrary"`
      if [ ${nphbw} -gt 0 ] || [ ${npasw} -gt 0 ]; then
        echo "<I> (AMEGIC) detected library writing: "${nphbw}" (HBW), "${npasw}" (ASW)"
        FLGWRITLIB="TRUE"
      else
        FLGWRITLIB="FALSE"
      fi

    done

  fi

###  cd ${shrun}
fi

if [ "${FLGAMEGIC}" == "TRUE" ]; then
## last pass (event generation)
  if [ "${lbo}" = "EVTS" ]; then
    echo " <I> generating events..."
    ${sherpaexe} "PATH="${pth} "PATH_PIECE="${pth} "RESULT_DIRECTORY="${dir2} 1>${shrun}/${outflbs}_passE.out 2>${shrun}/${outflbs}_passE.err
  fi
fi


## generate tar balls with data cards, libraries, cross sections, events
cd ${pth} ##Sherpa 1.3.0 needs full path MN 070611
#MM() cd ${pth}

#if [ "${lbo}" = "LBCR" ] || [ "${lbo}" = "CRSS" ]; then
#  mv ../Weights*.*      ./${dir2}/
#  mv ../xsections_*.dat ./${dir2}/
#fi

## libraries & cross sections
if [ "${lbo}" = "LIBS" ] || [ "${lbo}" = "LBCR" ]; then
  touch ${libsfile}.tmp
  find ./${dir1}/ -name '*'     > tmp.lst && tar --no-recursion -rf ${libsfile}.tmp -T tmp.lst; rm tmp.lst
  gzip -9 ${libsfile}.tmp && mv ${libsfile}.tmp.gz ${libsfile}
  mv ${libsfile} ${shrun}/
fi

if [ "${lbo}" = "LBCR" ] || [ "${lbo}" = "CRSS" ]; then
  touch ${crssfile}.tmp
  find ./${dir2}/ -name '*'     > tmp.lst && tar --no-recursion -rf ${crssfile}.tmp -T tmp.lst; rm tmp.lst
  if [ -e ${dir3} ]; then
  find ./${dir3}/ -name '*'     > tmp.lst && tar --no-recursion -rf ${crssfile}.tmp -T tmp.lst; rm tmp.lst
  fi
  gzip -9 ${crssfile}.tmp && mv ${crssfile}.tmp.gz ${crssfile}
  mv ${crssfile} ${shrun}/
fi

#### create tarball with multiple interactions grid files
migdir=`find ./ -type d -name MIG\*`
echo " <I> MPI grid located in "${migdir}
migfil=`find ./ -type f -name MPI\*.dat`
echo " <I> MPI file found: "${migfil}
if [ -d "${migdir}" ]; then
  if [ -e "${migfil}" ]; then
    tar -czf ${gridfile} ${migdir} ${migfil}
  else
    tar -czf ${gridfile} ${migdir}
  fi
  mv ${gridfile} ${shrun}/
fi
####

if [ "${lbo}" = "EVTS" ]; then
  rm ${libsfile}
  rm ${crssfile}
  rm ${gridfile}
  tar -czf ${evtsfile} *.*
  mv ${evtsfile} ${shrun}/
fi
#rm -rf ${dir1}/*
#rm -rf ${dir2}/*
#rm -rf ${dir3}/*
#rm *.md5

## data cards
FILES=`ls *.md5 *.dat *slha.out 2> /dev/null`
if [ "${lbo}" = "LIBS" ]; then
  tar -czf ${crdcfile} ${FILES}
  mv ${crdcfile} ${shrun}/
elif [ "${lbo}" = "LBCR" ]; then
  if [ "${FLGAMISIC}" = "TRUE" ]; then
    sed -e 's:MI_HANDLER.*:MI_HANDLER   = Amisic:' < Run.dat > Run.dat.tmp
    mv Run.dat.tmp Run.dat
  fi
  tar -czf ${crdefile} ${FILES}
  mv ${crdefile} ${shrun}/
elif [ "${lbo}" = "CRSS" ]; then
  if [ "${FLGAMISIC}" = "TRUE" ]; then
    sed -e 's:MI_HANDLER.*:MI_HANDLER   = Amisic:' < Run.dat > Run.dat.tmp
    mv Run.dat.tmp Run.dat
  fi
  tar -czf ${crdefile} ${FILES}
  mv ${crdefile} ${shrun}/
elif [ "${lbo}" = "EVTS" ]; then
  echo evts
fi

## log files
cd ${shrun}
FILES=`ls *.err *.out 2> /dev/null`
if [ "${lbo}" = "LIBS" ]; then
  tar -czf ${loglfile} ${FILES}
elif [ "${lbo}" = "LBCR" ]; then
  tar -czf ${loglfile} ${FILES}
elif [ "${lbo}" = "CRSS" ]; then
  tar -czf ${logcfile} ${FILES}
elif [ "${lbo}" = "EVTS" ]; then
  tar -czf ${logefile} ${FILES}
fi
#rm *.err *.out
mv *.tgz ${fin}/


# go back to original directory
cd ${HDIR}
rm -rf ${shrun}

