#!/bin/sh

function CreateHtml(){

    path=$1
    webadd="http://cmstac11.cern.ch:8080/analysis/"

    export webfile=$path/`echo $2 | sed -e "s@.txt@.html@g"`
    export htmlwebadd=`echo $webfile | sed -e "s@/data1/@$webadd@"`

    tableSeparator="<TD align=center>"

    rm -f $webfile
    echo "<TABLE  BORDER=1 ALIGN=CENTER> " > $webfile  

    cat $path/$2 | grep "|" | awk -F"|" '
BEGIN{
 Separator[0]="<TD align=center>"; 
 Separator[1]="</font><TD align=center>"; 
 fontColor[0]="<font color=\"#000000\">";
 fontColor[1]="<font color=\"#00000\">";
 ci=0;
 N=0;
}
{
  if (NR<2){
    label="";
    for (i=1;i<=NF;i++){
      label=sprintf("%s %s %s",label,Separator[0],$i);
    }      
    print label " <TR> ";
  }
  if (NR!=1){
       if (int($1)!=N) {N=int($1); if (ci==0){ci=1}else{ci=0} }
       pswebadd=""; label="<TD>"; 
       for (i=1;i<=NF;i++){
         val=$i;
         label=sprintf("%s %s %s %s",label,fontColor[0],val,Separator[1]);
        } 
       print label " <TR> " pswebadd;
  }
}' path=$path >> $webfile
    echo "</TABLE> " >> $webfile
}


####################################
#//////////////////////////////////#
####################################
##           LOCAL PATHS          ##
## change this for your local dir ##
####################################

## Where to find all the templates and to write all the logs
export LOCALHOME=/analysis/sw/CRAB
## Where to copy all the results
export MainStoreDir=/data1/CrabAnalysis
## Where to create crab jobs
export WorkingDir=/tmp/${USER}
## Leave python path as it is to source in standard (local) area
export python_path=/analysis/sw/CRAB

####################################
#//////////////////////////////////#
####################################

########################################
## Patch to make it work with crontab ##
###########################################
export MYHOME=/analysis/sw/CRAB
# the scritps will source ${MYHOME}/crab.sh
###########################################

theVersion=""
[ "$1" != "" ] && theVersion=$1

# Process all the Flags
for Version in `cat ${LOCALHOME}/Summary_cron.cfg | grep -v "#"`
  do
  [ "$theVersion" != "" ] && [ "$Version" != "$theVersion" ] && continue

  echo $Version

  logPath=${LOCALHOME}/log/${Version}
  CreatedRuns=`ls $logPath/Created`

  [ ! -e ${MainStoreDir}/ClusterAnalysis/${Version}/AllSummaries ] && mkdir ${MainStoreDir}/ClusterAnalysis/${Version}/AllSummaries
  echo -e "Run |\t CreationTime |\t SubmissionTime |\t StillWaitingNumber |\t DeltaSubmission |\t DeltaWaiting |\t StillWaitingNumber |\tmeanTime x good - meanTime x bad" > ${MainStoreDir}/ClusterAnalysis/${Version}/AllSummaries/CRABStatistic.txt
  echo -e " |\t day h m s  |\t day h m s |\t  day h m s |\t  |\t  |\t  |\t " >> ${MainStoreDir}/ClusterAnalysis/${Version}/AllSummaries/CRABStatistic.txt
  
  for run in $CreatedRuns
    do
 
    CreationTime=`grep "running on" $logPath/Created/$run | cut -d " " -f 9- | sed -e "s#:# #g" | awk '{day=$1;h=$2;m=$3;s=$4; print $0 "?" (((day*24)+h)*60+m)*60+s}'`
    SubmissionTime=`grep "running on" $logPath/Submitted/$run | cut -d " " -f 9- | sed -e "s#:# #g" | awk '{day=$1;h=$2;m=$3;s=$4; print $0 "?" (((day*24)+h)*60+m)*60+s}'`
    StillWaitingAtTime=`grep "running on" $logPath/Status/${run}.txt | cut -d " " -f 9- | sed -e "s#:# #g" | awk '{day=$1;h=$2;m=$3;s=$4; print $0 "?" (((day*24)+h)*60+m)*60+s}'`
    StillWaitingNumber=`grep -v ">>>>>>" $logPath/Status/${run}.txt | grep -c "Waiting"`
    StillScheduledNumber=`grep -v ">>>>>>" $logPath/Status/${run}.txt | grep -c "Scheduled"`
    let StillWaitingNumber=$StillWaitingNumber+$StillScheduledNumber
    TotalJobNumberNumber=`grep "Total Jobs" $logPath/Status/${run}.txt | cut -d " " -f 2 `

    T0=`echo $CreationTime   | cut -d "?" -f 2`
    T1=`echo $SubmissionTime | cut -d "?" -f 2`
    DeltaSubmission=`echo $T1 $T0 | awk '{delta=$1-$2; s=delta%60; m=int(delta/60); h=int(m/60); m=m%60; d=int(h/24); h=h%24; print d " " h":"m":"s }'`
    
    DeltaWaiting=0
    if [ "$StillWaitingNumber" != "0" ] ; then
	T0=`echo $SubmissionTime | cut -d "?" -f 2`
	T1=`echo $StillWaitingAtTime   | cut -d "?" -f 2`
	DeltaWaiting=`echo $T1 $T0 | awk '{delta=$1-$2; s=delta%60; m=int(delta/60); h=int(m/60); m=m%60; d=int(h/24); h=h%24; print d " " h":"m":"s}'`
    fi

    ExitCode=(`grep ExeExitCode ${MainStoreDir}/ClusterAnalysis/${Version}/$run/logs/*stdout 2>/dev/null | awk -F"=" '{print $2}'`)
    ExitTime=(`grep ExeTime ${MainStoreDir}/ClusterAnalysis/${Version}/$run/logs/*stdout 2>/dev/null | awk -F"=" '{print $2}'`)
    
    i=0
    meanG=0
    countG=0
    meanB=0
    countB=0
    while [ $i -lt   ${#ExitCode[@]} ];
      do
      [ ${ExitCode[$i]} -eq 0 ] && let meanG=${meanG}+${ExitTime[$i]} && let countG++
      [ ${ExitCode[$i]} -ne 0 ] && let meanB=${meanB}+${ExitTime[$i]} && let countB++
      let i++
    done
    
    [ $countG -ne 0 ] &&  let meanG=$meanG/$countG
    [ $countB -ne 0 ] && let meanB=$meanB/$countB
    meanG=`echo $meanG | awk '{delta=$1; s=delta%60; m=int(delta/60); h=int(m/60); m=m%60; d=int(h/24); h=h%24; print d " " h":"m":"s}'`
    meanB=`echo $meanB | awk '{delta=$1; s=delta%60; m=int(delta/60); h=int(m/60); m=m%60; d=int(h/24); h=h%24; print d " " h":"m":"s}'`
#    echo meanG $meanG
#    echo meanB $meanB
    
    echo -e "$run  | `echo $CreationTime | cut -d '?' -f 1` |\t `echo $SubmissionTime | cut -d '?' -f 1` |\t `echo $StillWaitingAtTime | cut -d '?' -f 1` |\t $DeltaSubmission |\t $DeltaWaiting |\t $StillWaitingNumber / $TotalJobNumberNumber |\t $meanG - $meanB" >> ${MainStoreDir}/ClusterAnalysis/${Version}/AllSummaries/CRABStatistic.txt
    
  done

  CreateHtml ${MainStoreDir}/ClusterAnalysis/${Version}/AllSummaries CRABStatistic.txt
done
