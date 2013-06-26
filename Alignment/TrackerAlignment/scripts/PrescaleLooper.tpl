#! /bin/bash
#set -x

source /afs/cern.ch/cms/caf/setup.sh
curdir=$( pwd )

workdir="<MYCMSSW>/src/"
CASTOR_OUT="<CASTOROUT>"

DQM_OUT="${curdir}/MONITORING/DQM/CTF/"
#DQM_OUT=$1 
ALCAFILELIST=$1
MAXEVENTS=18000


checkCorruptedFiles(){


    FOUND=4 #not found by default 
    if [ $# != 1 ]
	then
#	echo "wrong number of input parameters ( $# ). Please provide a number (index of job to check). Exiting with error."
	echo 3
    fi
 
    LISTBADJOBS="/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERALIGN2/HIP/bonato/CMSSW_3_8_4_patch2/src/Alignment/TrackerAlignment/scripts/badjobs_Run2010A-v4.lst" 
  #  echo "Called checkCorruptedFiles with input $1"

    IND=$1
   while read myline

      do
#      echo "LINE is $myline" 
#      VAR1=$( echo $myline | awk '{print $1}' )
#      VAR2=$( echo $VAR1 | sed 's/^M$//' )
      if [[ "$1" -eq "$myline" ]]
	  then
#	echo "Found INDEX $1"
	FOUND=0
	  break
    #  else
	#  echo "INDEX $1 !!== $myline"
      fi
    done  < $LISTBADJOBS
    
    echo $FOUND
    
}
# end checkCorruptedFiles()



########################
### START MAIN BODY OF THE SCRIPT

#check if output directory exists
nsls -d /castor/cern.ch/cms/$CASTOR_OUT
if [ $? -ne 0 ]
then
echo "Output directory: "
echo $CASTOR_OUT
echo "does not exist. Please check the scripts. Exiting."
exit 1
fi


echo ""
echo "I am in $curdir"

cnt=0;

if [ $# -gt 1 ]
    then  
    if [ $2 -gt 2 ]
	then
	echo "Cleaning from Prescaled files the output directory: /castor/cern.ch/cms/$CASTOR_OUT"  
	for file in $(nsls /castor/cern.ch/cms/$CASTOR_OUT/ | grep 'Prescaled') 
	  do
#echo "deleting $file"
	  rfrm  /castor/cern.ch/cms/$CASTOR_OUT/$file
	  let cnt=cnt+1
	done
	echo "Deleted $cnt files"
    fi
fi


cd $workdir
eval `scramv1 runtime -sh`
export STAGE_SVCCLASS=cmscaf
cd -

for ALCATAG in $( cat $ALCAFILELIST  )
do

  echo
  echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
  echo "%%% Starting the ALCATAG: ${ALCATAG}"
  echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
  echo
  ###DAT_FILE="/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERALIGN/HIP/bonato/DEVEL/HIPWorkflow/ALCARECOskim/v1.4/data/${ALCATAG}.dat"
  DAT_FILE="${curdir}/../data/${ALCATAG}.dat"
  TPL_FILE="TkAlCaRecoPrescaling.${ALCATAG}.tpl"
  BASE_TPL=$(basename "$TPL_FILE" .tpl)
  TAG=$ALCATAG #"CRAFT" 
  JOBTAG="ALCAPresc_"${TAG}

#produce list of the DQM files produced at the Skim loop
cd $DQM_OUT
dqmlist=${DQM_OUT}"/AlignmentDQMHitMaps_CTF_${TAG}.txt"
rm -f $dqmlist
for dqmfile in  $(ls  TkAl*Skim*${TAG}*HitMaps*.root)
do
echo $(pwd)/$dqmfile >> $dqmlist
done

#run the merging of DQM and calculation of the prescaling factors.
#It is fast, do it locally
cd $curdir
dqmtotfile="${curdir}/TkAlDQMHitMaps_CTF_${TAG}.root"
#replace "<DQMLIST>" $dqmlist "<DQMTOTFILE>" $dqmtotfile  < mergemytree_cfg.tpl > mergemytree_${TAG}_cfg.py
sed -e "s|<DQMLIST>|${dqmlist}|g"  -e "s|<DQMTOTFILE>|${dqmtotfile}|g"   < mergemytree_cfg.tpl > mergemytree_${TAG}_cfg.py
echo "Merging DQM into $dqmtotfile"
time cmsRun mergemytree_${TAG}_cfg.py



#repeat procedure of SkimLooper.sh, splitting the data sample in even bunches of events
  TOTFILES=0
  INDEX=1
  for i in $( cat $DAT_FILE )
    do

#pick the total nr events in this file from the previously produced nevents.out
    let TOTFILES=TOTFILES+1
    TOTEVTS=$(sed -n $TOTFILES'p' ../data/nevents${TAG}.out)
#echo "The file #$TOTFILES has $TOTEVTS events"
    if [ $TOTEVTS == 0 ]
	then
#	echo "The file #$TOTFILES has $TOTEVTS events"
	continue
    fi 
    TOTSPLITS=$(( ( $TOTEVTS / $MAXEVENTS ) +1 ))
    firstev=0
    lastev=-1
    
#echo "I will split it into $TOTSPLITS"
    if [ $TOTSPLITS > 1 ]
	then
	nsplits=1

#prepare configs for prescaling
	while [ $nsplits -le $TOTSPLITS  ]
	  do
#echo "Splitting the file $TOTFILE : $nsplits"
	  firstev=$(( $MAXEVENTS*$(( $nsplits-1 ))+1 ))
	  lastev=$MAXEVENTS    #$(( ($MAXEVENTS*$nsplits) ))
	  JOB=$JOBTAG"_file"$INDEX 
	  CFG_FILE=$BASE_TPL"."$TAG"_cfg."$INDEX".py"
	  sed -e "s|<JOB>|${JOB}|g"  -e "s|<INPATH>|${i}|g"   -e "s|<INIEVT>|${firstev}|g"  -e "s|<FINEVT>|${lastev}|g"  -e "s|<ALCATAG>|${ALCATAG}|g"  -e "s|<MERGEDHITMAP>|${dqmtotfile}|g"  < $TPL_FILE > $CFG_FILE
	  let INDEX=INDEX+1
	  let nsplits=nsplits+1
# 	  if [ $INDEX -ge 3 ]
# 	      then
# 	      echo "Reached a maximum number of files: $INDEX. Stopping the submission"
# 	      break
# 	  fi

	done

    else #file is small and does not contain too many events
	firstev=0
	lastev=-1
	JOB=$JOBTAG"_file"$INDEX 
	CFG_FILE=$BASE_TPL"."$TAG"_cfg."$INDEX".py"
	sed -e "s|<JOB>|${JOB}|g"  -e "s|<INPATH>|${i}|g"   -e "s|<INIEVT>|${firstev}|g"  -e "s|<FINEVT>|${lastev}|g"  -e "s|<ALCATAG>|${ALCATAG}|g"  -e "s|<MERGEDHITMAP>|$dqmtotfile|g"  < $TPL_FILE > $CFG_FILE
	let INDEX=INDEX+1
# 	if [ $INDEX -ge 3 ]
# 	    then
# 	    echo "Reached a maximum number of files: $INDEX. Stopping the submission"
# 	    break
# 	fi

    fi


#     if [ $INDEX -ge 3 ]
# 	then
# 	echo "Reached a maximum number of files: $INDEX. Stopping the submission"
# 	break
#     fi

#echo "--- moving to next file. At the moment INDEX=$INDEX"
  done
  


TOTPRESCALEDJOBS=$(( $INDEX -1 ))
echo 
echo "TOTPRESCALEDJOBS = $TOTPRESCALEDJOBS"


#submit them
echo
echo
echo "@@@@@@@@@@@@@@@@@@@@@"
echo
INDEX=1
while [ $INDEX -le $TOTPRESCALEDJOBS ]
do
JOBNAME="ALCAPresc"$TAG"_"$INDEX
LOGFILE="${JOBNAME}.log"
CFG_FILE=$BASE_TPL"."$TAG"_cfg."$INDEX".py"


CHECKCORRUPTED=1 # if greater than zero it overrides the checks
FILECORRUPTED=$( checkCorruptedFiles $INDEX )


if [[ $FILECORRUPTED == 0 || CHECKCORRUPTED -gt 0 ]]
then
    echo "Submitting $JOBNAME with config file $CFG_FILE"
    REM=0
    let "REM=$INDEX % 300"
    if [ $REM -lt 100 ]
	then 
#echo "dummy D" > /dev/null
	bsub -q cmscaf1nd -J $JOBNAME -oo $LOGFILE presc_exec.sh "$curdir/$CFG_FILE" "$CASTOR_OUT"  "$DQM_OUT"
    elif [ $REM -lt 200 ] 
	then
	#echo "dummy E" > /dev/null
	bsub -q cmsexpress -J $JOBNAME -oo $LOGFILE presc_exec.sh "$curdir/$CFG_FILE" "$CASTOR_OUT"  "$DQM_OUT"
    else
	#echo "dummy F" > /dev/null
	bsub -q cmscaf1nd    -J $JOBNAME -oo $LOGFILE presc_exec.sh "$curdir/$CFG_FILE" "$CASTOR_OUT"  "$DQM_OUT"
    fi

fi #end if file is corrupted

let INDEX=INDEX+1
done #end while loop on submissions


done #end for loop on TAG list
