#!/bin/bash
#
#  file:        MakeSherpaLibs.sh
#  description: BASH script handling the SHERPA data cards for the
#               library production and cross section calculation
#  uses:        the required SHERPA data cards (+ libraries) [see below]
#
#  author:      Markus Merschmeyer, RWTH Aachen
#  date:        2008/05/13
#  version:     1.5
#



# +-----------------------------------------------------------------------------------------------+
# function definitions
# +-----------------------------------------------------------------------------------------------+

print_help() {
    echo "" && \
    echo "MakeSherpaLibs version 1.5" && echo && \
    echo "options: -r  path       path to your SHERPA installation" && \
    echo "                         -> ( "${shr}" )" && \
    echo "         -p  process    SHERPA process/dataset name ( "${prc}" )" && \
    echo "         -o  options    library/cross section options ( "${lbo}" )" && \
    echo "                         [ 'LBCR' : generate libraries and cross sections     ]" && \
    echo "                         [ 'LIBS' : generate libraries only                   ]" && \
    echo "                         [ 'CRSS' : generate cross sections, needs libraries! ]" && \
    echo "                         [ 'EVTS' : generate events, needs libs + crss. sec.! ]" && \
    echo "         -i  path       path to SHERPA datacard (and library, see -o) files" && \
    echo "                         -> ( "${inc}" )" && \
    echo "         -D  filename   (optional) name of data card file ( "${cfdc}" )" && \
    echo "         -L  filename   (optional) name of library file ( "${cflb}" )" && \
    echo "         -C  filename   (optional) name of cross section file ( "${cfcr}" )" && \
    echo "         -f  path       output path for SHERPA library & cross section files" && \
    echo "                         -> ( "${fin}" )" && \
    echo "         -d  directory  (optional) name of SHERPA 'Run' subdirectory ( "${pth}" )" && \
    echo "         -h             display this help and exit" && echo
}

check_md5() {
# $1 : what is to be checked ('CRDS','CRDFILE','LIBS''LIBFILE','CRSS','CRSFILE') ?
# $2 : who is requesting the check ('CRDS','LIBS','CRSS','EVTS') ?
# $3 : name of the file containing the checksums
IMD=`which md5sum | grep -c -i "not found"`
if [ $IMD -eq 0 ]; then
  if [ -e $3 ]; then
    rslt=`md5sum --check $3`
    fpatt="OK"
    nok=`echo $rslt | grep -o -i $fpatt | grep -c -i $fpatt`
    fpatt="FAILED"
    nfail=`echo $rslt | grep -o -i $fpatt | grep -c -i $fpatt`
    nline=`cat $3 |wc -l`
    echo " <I> MD5 file has "$nline" entries"
    echo " <I>  -> OK: "$nok", FAILED: "$nfail
    if [ $nfail -gt 0 ]; then
      if [ "$1" = "CRDS" ] && [ "$2" = "CRDS" ]; then
        echo " <E> data cards do not match their own MD5 sums,"
        echo " <E>  stopping..."
        exit 1
      fi
      if [ "$1" = "CRDFILE" ] && [ "$2" = "LIBS" ]; then
        echo " <W> libraries were probably not generated with this data card file"
      fi
      if [ "$1" = "CRDS" ] && [ "$2" = "LIBS" ]; then
        echo " <W> some of the data cards have changed since library generation"
        echo " <W>  please make sure these changes are harmless:"
        cnt=1
        while [ $cnt -le $nline ]; do
          let fidx=$cnt+1
          cfile=`echo $rslt |cut -f $cnt -d ":" | cut -f 3 -d " "`
          cstat=`echo $rslt |cut -f $fidx -d ":" | cut -f 2 -d " "`
          if [ "$cstat" = "FAILED" ]; then
            echo " <W>  -> file '"$cfile"' does not match MD5 sum"
          fi
          let cnt=$cnt+1
        done
      fi
      if [ "$1" = "LIBS" ] && [ "$2" = "LIBS" ]; then
        echo " <E> libraries do not match their own MD5 sums,"
        echo " <E>  stopping..."
        exit 1
      fi
    fi
  else
    echo " <W> file "$3" does not exist, skipping MD5 test"
  fi # check existence od md5sum file
fi   # check availability of 'md5sum'
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
    cd ../..
  done
}



# +-----------------------------------------------------------------------------------------------+
# start of the script
# +-----------------------------------------------------------------------------------------------+

# save current path
HDIR=`pwd`

# dummy setup (if all options are missing)
shr=${HDIR}/SHERPA-MC-1.1.1        # path to SHERPA installation
pth="LHC"                          # name of SHERPA data card directory
prc="XXX"                          # SHERPA process name
lbo="LBCR"                         # library/cross section option
inc=${HDIR}                        # path to SHERPA datacards (libraries)
cfdc=""                            # custom data card file name
cflb=""                            # custom library file name
cfcr=""                            # custom cross section file name
fin=${HDIR}                        # output path for SHERPA libraries & cross sections

# get & evaluate options
while getopts :r:d:p:o:i:D:L:C:f:h OPT
do
  case $OPT in
  r) shr=$OPTARG ;;
  d) pth=$OPTARG ;;
  p) prc=$OPTARG ;;
  o) lbo=$OPTARG ;;
  i) inc=$OPTARG ;;
  D) cfdc=$OPTARG ;;
  L) cflb=$OPTARG ;;
  C) cfcr=$OPTARG ;;
  f) fin=$OPTARG ;;
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
cd ${shr}; shr=`pwd`; cd ${HDIR}
cd ${inc}; inc=`pwd`; cd ${HDIR}
cd ${fin}; fin=`pwd`; cd ${HDIR}

# find 'Run' directory
cd ${shr}
shdir=`ls | grep "SHERPA"`
echo " <I> SHERPA directory is: "${shdir}
cd -
shrun=${shr}/${shdir}/Run

echo "  -> SHERPA path: '"${shr}"'"
echo "  -> SHERPA data card directory: '"${pth}"'"
echo "  -> PROCESS name: '"${prc}"'"
echo "  -> Library & cross section otions: '"${lbo}"'"
echo "  -> include path: '"${inc}"'"
echo "  -> custom data card file name: '"${cfdc}"'"
echo "  -> custom library file name: '"${cflb}"'"
echo "  -> custom cross section file name: '"${cfcr}"'"
echo "  -> output path: '"${fin}"'"


### go to 'Run' subdirectory of SHERPA
cd ${shrun}


### set base name for SHERPA output file(s) and directories
outflbs=sherpa_${prc}
if [ "${cfdc}" = "" ]; then
  cardfile=${outflbs}_cards.tgz           # input card file (master -> libraries)
else
  cardfile=${cfdc}                        # custom input data card file
  echo " <I> using custom data card file: "${cardfile}
fi
crdsmd5s=md5sums_crds.md5                 # MD5 sums -> cards
crdfmd5s=md5sums_crdsfile.md5             # MD5 sum -> cardfile
if [ "${cflb}" = "" ]; then
  libsfile=${outflbs}_libs.tgz            # output libs
else
  libsfile=${cflb}                        # custom input library file
  echo " <I> using custom library file: "${libsfile}
fi
libsmd5s=md5sums_libs.md5                 # MD5 sums -> libs
libfmd5s=md5sums_libsfile.md5             # MD5 sum -> libfile
if [ "${cfcr}" = "" ]; then
  crssfile=${outflbs}_crss.tgz            # output cross sections
else
  crssfile=${cfcr}                        # custom input cross section file
  echo " <I> using custom cross section file: "${crssfile}
fi
crssmd5s=md5sums_crss.md5                 # MD5 sums -> cross sections
crsfmd5s=md5sums_crssfile.md5             # MD5 sum -> cross section file
evtsfile=${outflbs}_evts.tgz              # output events
evtsmd5s=md5sums_evts.md5                 # MD5 sums -> events
evtfmd5s=md5sums_evtsfile.md5             # MD5 sum -> eventfile
#
crdlfile=${outflbs}_crdL.tgz              # output cardfile (-> from library production)
crdcfile=${outflbs}_crdC.tgz              # output cardfile (-> from cross section calculation)
crdefile=${outflbs}_crdE.tgz              # output cardfile (-> from event generation)
#
loglfile=${outflbs}_logL.tgz              # output messages (-> from library production)
logcfile=${outflbs}_logC.tgz              # output messages (-> from cross section calculation)
logefile=${outflbs}_logE.tgz              # output messages (-> from event generation)
#
#if [ "${lbo}" = "LIBS" ] || [ "${lbo}" = "LBCR" ]; then
#elif [ "${lbo}" = "CRSS" ]; then
#fi
#
dir1="Process"
dir2="Result"


### clean up existing xsection files
for file in `ls xsections_*.dat`; do
  echo " <W> deleting existing cross section file: "${file}
  rm ${file}
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


### get data card (+ library) tarball(s) from include path
cp ${inc}/${cardfile} .
if [ "${lbo}" = "CRSS" ]; then
  cp ${inc}/${libsfile} .
fi
if [ "${lbo}" = "EVTS" ]; then
  cp ${inc}/${libsfile} .
  cp ${inc}/${crssfile} .
fi


### check existence of data card file
if [ -e ${cardfile} ]; then
  echo " data card file '"${cardfile}"' exists,"
  fsize=`ls -l ${cardfile} | awk '{print $5}'`
  if [ ${fsize} -gt 0 ]; then
    echo "  -> unpacking data card file"
    mv  ${cardfile} ${pth}/
    cd ${pth}
    tar -xzvf ${cardfile}
    check_md5 "CRDS" "CRDS" ${crdsmd5s}
    cd -
  else
    echo " <E> file "${cardfile}" is empty"
    echo "  -> stopping..." 
    exit
  fi
else
  echo " <E> no data card file found"
  echo "  -> stopping..."
  exit
fi


### check required subdirectories
## generate/clean 'Process' subdirectory
if [ ! -e ${pth}/${dir1} ]; then
  echo " '"${pth}/${dir1}"' subdirectory does not exist and will be created"
  mkdir ${pth}/${dir1}
else
  echo " cleaning '"${pth}/${dir1}"' subdirectory"
  rm -rf ${pth}/${dir1}/*
fi
## generate/clean 'Result' subdirectory
if [ ! -e ${pth}/${dir2} ]; then
  echo " '"${pth}/${dir2}"' subdirectory does not exist and will be created"
  mkdir ${pth}/${dir2}
else
  echo " cleaning '"${pth}/${dir2}"' subdirectory"
  rm -rf ${pth}/${dir2}/*
fi


### check, whether only cross sections have to be calculated
if [ "${lbo}" = "CRSS" ] || [ "${lbo}" = "EVTS" ]; then
  if [ -e ${libsfile} ]; then
    echo " <I> library file '"${libsfile}"' exists,"
    fsize=`ls -l ${libsfile} | awk '{print $5}'`
    if [ ${fsize} -gt 0 ]; then
      echo "  -> unpacking library file and cleaning '"${dir2}"' subdirectory"
      mv ${libsfile} ${pth}/
      cd ${pth}
      tar -xzf ${libsfile}
      check_md5 "CRDFILE" "LIBS" ${crdfmd5s}
#      rm ${cardfile}
      check_md5 "CRDS"    "LIBS" ${crdsmd5s}
      cd -
    else
      echo " <E> file "${libsfile}" is empty"
      echo "  -> stopping..." 
      exit
    fi
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
    fsize=`ls -l ${crssfile} | awk '{print $5}'`
    if [ ${fsize} -gt 0 ]; then
      echo "  -> unpacking cross section file and cleaning '"${dir2}"' subdirectory"
      mv ${crssfile} ${pth}/
      cd ${pth}
      tar -xzf ${crssfile}
      check_md5 "CRDFILE" "CRSS" ${crdfmd5s}
      check_md5 "CRDS"    "CRSS" ${crdsmd5s}
      check_md5 "LIBFILE" "CRSS" ${libfmd5s}
      check_md5 "LIBS"    "CRSS" ${libsmd5s}
      cd -
    else
      echo " <E> file "${crssfile}" is empty"
      echo "  -> stopping..." 
      exit
    fi
  else
    echo " <E> no cross section file found"
    echo "  -> stopping..."
    exit
  fi
fi



### generate process-specific libraries -> redirect output (stdout, stderr) to files
## first pass
cp ${shr}/bin/Sherpa ${shrun}/
if [ "${lbo}" = "LIBS" ] || [ "${lbo}" = "LBCR" ]; then
  echo " <I> creating library code..."
  ./Sherpa "PATH="${pth} "RESULT_DIRECTORY="${pth}/${dir2} 1>${shrun}/${outflbs}_pass1.out 2>${shrun}/${outflbs}_pass1.err
  cd ${pth}
  cp ${shr}/share/SHERPA-MC/makelibs .
  echo " <I> compiling libraries..."
  ./makelibs 1>${shrun}/${outflbs}_mklib.out 2>${shrun}/${outflbs}_mklib.err
  lsize=`du -sh  | cut -f 1 -d "."`
  echo " <I>  -> raw size: "${lsize}
  echo " <I> cleaning libraries..."
  clean_libs 1>${shrun}/${outflbs}_cllib.out 2>${shrun}/${outflbs}_cllib.err
  lsize=`du -sh  | cut -f 1 -d "."`
  echo " <I>  -> clean size: "${lsize}
  cd ${shrun}
fi
## second pass (save integration results)
if [ "${lbo}" = "LBCR" ] || [ "${lbo}" = "CRSS" ]; then
  echo " <I> calculating cross sections..."
  ./Sherpa "PATH="${pth} "RESULT_DIRECTORY="${pth}/${dir2} 1>${shrun}/${outflbs}_pass2.out 2>${shrun}/${outflbs}_pass2.err
fi
## third pass (event generation)
if [ "${lbo}" = "EVTS" ]; then
  echo " <I> generating events..."
  ./Sherpa "PATH="${pth} "RESULT_DIRECTORY="${pth}/${dir2} 1>${shrun}/${outflbs}_pass3.out 2>${shrun}/${outflbs}_pass3.err
fi


## generate tar balls with data cards, libraries, cross sections, events
cd ${shrun}/${pth}
## data cards
md5sum *.dat  >  ${crdsmd5s}
md5sum *.slha >> ${crdsmd5s}
if [ "${lbo}" = "LIBS" ]; then
  tar -czf ${crdlfile} *.md5 *.dat *.slha
  md5sum ${crdlfile} > ${crdfmd5s}
elif [ "${lbo}" = "LBCR" ]; then
  mv ../xsections_*.dat ./
  tar -czf ${crdlfile} *.md5 *.dat *.slha
  md5sum ${crdlfile} > ${crdfmd5s}
elif [ "${lbo}" = "CRSS" ]; then
  mv ../xsections_*.dat ./
  tar -czf ${crdcfile} *.md5 *.dat *.slha
  md5sum ${crdcfile} > ${crdfmd5s}
elif [ "${lbo}" = "EVTS" ]; then
  mv ../xsections_*.dat ./
  tar -czf ${crdefile} *.md5 *.dat *.slha
  md5sum ${crdefile} > ${evtfmd5s}
fi
rm *.dat
if [ -e ${crdlfile} ]; then
  mv ${crdlfile} ${shrun}/
fi
if [ -e ${crdcfile} ]; then
  mv ${crdcfile} ${shrun}/
fi
if [ -e ${crdefile} ]; then
  mv ${crdefile} ${shrun}/
fi
## libraries
if [ "${lbo}" = "LIBS" ]; then
  md5sum ${dir1}/lib/*.* > ${libsmd5s}
  tar -czf ${libsfile} *.md5 ${dir1}/*
  md5sum ${libsfile} > ${libfmd5s}
elif [ "${lbo}" = "LBCR" ]; then
  md5sum ${dir1}/lib/*.* > ${libsmd5s}
  tar -czf ${libsfile} *.md5 ${dir1}/*
  md5sum ${libsfile} > ${libfmd5s}
  md5sum ${dir2}/*.* > ${crssmd5s}
  tar -czf ${crssfile} *.md5 ${dir2}/*
  md5sum ${crssfile} > ${crsfmd5s}
elif [ "${lbo}" = "CRSS" ]; then
  md5sum ${dir1}/lib/*.* > ${libsmd5s}
  md5sum ${libsfile} > ${libfmd5s}
  rm ${libsfile}
  md5sum ${dir2}/*.* > ${crssmd5s}
  tar -czf ${crssfile} *.md5 ${dir2}/*
  md5sum ${crssfile} > ${crsfmd5s}
elif [ "${lbo}" = "EVTS" ]; then
  md5sum ${dir1}/lib/*.* > ${libsmd5s}
  md5sum ${libsfile} > ${libfmd5s}
  rm ${libsfile}
  md5sum ${dir2}/*.* > ${crssmd5s}
  md5sum ${crssfile} > ${crsfmd5s}
  rm ${crssfile}
  md5sum *.*         > ${evtsmd5s}
#  tar -czf ${evtsfile} *.md5 *.*
  tar -czf ${evtsfile} *.*
  md5sum ${evtsfile} > ${evtfmd5s}
fi
if [ -e ${libsfile} ]; then
  mv ${libsfile} ${shrun}/
fi
rm -rf ${dir1}/*
if [ -e ${crssfile} ]; then
  mv ${crssfile} ${shrun}/
fi
if [ -e ${evtsfile} ]; then
  mv ${evtsfile} ${shrun}/
fi
rm -rf ${dir2}/*
rm *.md5
## log files
cd ${shrun}
if [ "${lbo}" = "LIBS" ]; then
  tar -czf ${loglfile} *.err *.out
elif [ "${lbo}" = "LBCR" ]; then
  tar -czf ${loglfile} *.err *.out
elif [ "${lbo}" = "CRSS" ]; then
  tar -czf ${logcfile} *.err *.out
elif [ "${lbo}" = "EVTS" ]; then
  tar -czf ${logefile} *.err *.out
fi
rm *.err *.out
mv *.tgz ${fin}/


# go back to original directory
cd ${HDIR}

