#! /bin/bash


curdir=$( pwd )

#workdir="/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERALIGN/HIP/bonato/DEVEL/HIPWorkflow/CMSSW_3_2_4/src/"
#CASTOR_OUT="/castor/cern.ch/cms/store/user/bonato/CRAFTReproSkims/Craft09/4T/"

workdir="<MYCMSSW>/src/"
CASTOR_OUT="<CASTOROUT>"

##DQM_OUT="/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERALIGN/HIP/bonato/DEVEL/HIPWorkflow/ALCARECOskim/v1.4/MONITORING/DQM/CTF/"
DQM_OUT="${curdir}/MONITORING/DQM/CTF/"
#DQM_OUT=$1 
ALCAFILELIST=$1
MAXEVENTS=18000
BASE_TPL=$(basename "$TPL_FILE" .tpl)

#check if output directory exists
nsls $CASTOR_OUT
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
	echo "Cleaning from Prescaled files the output directory: $CASTOR_OUT"  
	for file in $(nsls $CASTOR_OUT/ | grep 'Prescaled') 
	  do
#echo "deleting $file"
	  rfrm  $CASTOR_OUT/$file
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
  DAT_FILE="${curdir}/data/${ALCATAG}.dat"
  TPL_FILE="TkAlCaRecoPrescaling.${ALCATAG}.tpl"
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
replace "<DQMLIST>" $dqmlist "<DQMTOTFILE>" $dqmtotfile  < mergemytree_cfg.tpl > mergemytree_${TAG}_cfg.py
echo "Merging DQM into $dqmtotfile"
time cmsRun mergemytree_${TAG}_cfg.py



#repeat procedure of SkimLooper.sh, splitting the data sample in even bunches of events
  TOTFILES=0
  INDEX=1
  for i in $( cat $DAT_FILE )
    do

#pick the total nr events in this file from the previously produced nevents.out
    let TOTFILES=TOTFILES+1
    TOTEVTS=$(sed -n $TOTFILES'p' ./data/nevents${TAG}.out)
#echo "The file #$TOTFILES has $TOTEVTS events"

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
	  replace "<JOB>" $JOB "<INPATH>"  $i  "<INIEVT>" $firstev "<FINEVT>" $lastev "<ALCATAG>" $ALCATAG "<MERGEDHITMAP>" $dqmtotfile < $TPL_FILE > $CFG_FILE
	  let INDEX=INDEX+1
	  let nsplits=nsplits+1
	done

    else #file is small and does not contain too many events
	firstev=0
	lastev=-1
	JOB=$JOBTAG"_file"$INDEX 
	CFG_FILE=$BASE_TPL"."$TAG"_cfg."$INDEX".py"
	replace "<JOB>" $JOB "<INPATH>"  $i  "<INIEVT>" $firstev "<FINEVT>" $lastev  "<ALCATAG>" $ALCATAG "<MERGEDHITMAP>" $dqmtotfile< $TPL_FILE > $CFG_FILE
	let INDEX=INDEX+1
    fi

#echo "--- moving to next file. At the moment INDEX=$INDEX"
  done
  


TOTPRESCALEDJOBS=$(( $INDEX -1 ))
echo 
echo "TOTPRESCALEDJOBS = $TOTPRESCALEDJOBS"


#submit them
INDEX=1
while [ $INDEX -le $TOTPRESCALEDJOBS ]
do
JOBNAME="ALCAPresc"$TAG"_"$INDEX
LOGFILE="${JOBNAME}.log"
CFG_FILE=$BASE_TPL"."$TAG"_cfg."$INDEX".py"
echo "Submitting $JOBNAME with config file $CFG_FILE"
if [ $INDEX -lt 100 ]
then 
#echo "dummy D"
bsub -q cmscaf1nd -J $JOBNAME -oo $LOGFILE presc_exec.sh "$curdir/$CFG_FILE" "$CASTOR_OUT"  "$DQM_OUT"
elif [ $INDEX -lt 200 ] 
then
#echo "dummy E"
bsub -q cmsexpress -J $JOBNAME -oo $LOGFILE presc_exec.sh "$curdir/$CFG_FILE" "$CASTOR_OUT"  "$DQM_OUT"
else
#echo "dummy F"
bsub -q cmscaf1nh    -J $JOBNAME -oo $LOGFILE presc_exec.sh "$curdir/$CFG_FILE" "$CASTOR_OUT"  "$DQM_OUT"
fi

let INDEX=INDEX+1
done #end while loop on submissions


done #end for loop on TAG list
