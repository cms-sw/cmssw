#! /bin/zsh
HOMEDIR=$(dirname $0)
EMAIL=$1
TMPDIR=/tmp
AGENTS=("fileCollector" "producerFileCleanner")
LOGDIR=/home/dqmprolocal/agents
STOPFILE=/tmp/stopModules

######################################################################
# Support functions
startAgents(){
  [[ $1 == "all" ]] && agents=($AGENTS ) ||
      agents=$1

  for a in ${agents[@]}
  do  
    case $a in
      "fileCollector" )
        (set -x
         ($HOMEDIR/fileCollector.py lilopera@cern.ch \
                /home/dqmprolocal/output \
                /home/dqmprolocal/done  \
                /dqmdata/dqm/uploads
         ) |& $HOMEDIR/visDQMRotateLogs $LOGDIR/fcollect-%Y%m%d%H%M.txt </dev/null 86400 &
        )
        ;;
         
      "producerFileCleanner" )
        (set -x
         ($HOMEDIR/producerFileCleanner.py lilopera@cern.ch \
          /home/dqmprolocal/done \
          /home/dqmprolocal/output \
          /dqmdata/dqm/repository/original
         ) |& $HOMEDIR/visDQMRotateLogs $LOGDIR/pfclean-%Y%m%d%H%M.txt </dev/null 86400 & 
        )
        ;;
    esac 
  done
}

killproc() {
  local T title pat nextmsg
  T=1 title="$1" pat="$2"
  nextmsg="INFO: Stopping ${title}:"
  for pid in $(pgrep -u $(id -u) -f "$pat" | sort -rn); do
    psline=$(ps -o pid=,bsdstart=,args= $pid |
             perl -n -e 'print join(" ", (split)[0..4])')
    [ -n "$nextmsg" ] && { echo "$nextmsg"; nextmsg=; }
    echo -n "Stopping $pid ($psline):"
    for sig in TERM TERM QUIT KILL; do
      echo -n " SIG$sig"
      kill -$sig $pid
      sleep 1
      [ $(ps h $pid | wc -l) = 0 ] && break
      sleep $T
      T=$(expr $T \* 2)
      [ $(ps h $pid | wc -l) = 0 ] && break
    done
    echo
    newline="\n"
  done
}

logme(){
  timeTag=$(date +"%Y%m%d%H%M")
  logFile=$LOGDIR/alivecheck-${timeTag}
  if [[ ${#*} -eq 0 ]]
  then 
    while read a 
    do
      echo $(date +"%Y-%m-%d %H:%M:%S")" [aliveCheck.sh/$$] $a" >>! $logFile 
    done
  else
    echo $(date +"%Y-%m-%d %H:%M:%S")" [aliveCheck.sh/$$] $*" >>! $logFile
  fi
  return 0
}

######################################################################
# Setting up the environment
mkdir -p $LOGDIR
cd ~
if [[ -d prod && -e bin/setup_cmssw.sh ]] 
then
  cd prod
  source bin/setup_cmssw.sh
  eval `scramv1 runtime -sh`
  cd ~
else
  logme "ERROR: Could not find prod release of CMSSW please make sure" \
        "that ~/prod exists and is a symbolic link (ln -s) to the" \
        "CMSSW area used by dqmpro and the online consumers." \
        "Also make sure that ~/bin/setup_cmssw.sh points to the right" \
        "CMSSW installation area on nfs"
  exit
fi

if [[ -z $EMAIL || ! $EMAIL == *[a-zA-Z0-9\-\_\.]@cern.ch ]]
then 
  logme "ERROR: missing or unacceptable email address '$EMAIL'. e.g. " \
        "$HOMEDIR/aliveCheck.sh yourEmail@cern.ch"
  exit
fi
   
# Stop mode
if [ -e $STOPFILE ]
then
  logme "INFO: Found stop file (${STOPFILE}) at $HOSTNAME. Please" \
        "remove the file to restart the agents"
  set -a runningAgents
  for a in $AGENTS
  do
    pgrep -f $a > /dev/null && runningAgents[$(( ${#runningAgents} + 1 ))]=$a
  done
  for a in $runningAgents
  do 
    killproc "FMS Module [$a] " $a | logme
  done
  exit
fi

# Running Mode:
# Find out if there's any dead agents
set -a deadAgents
for a in $AGENTS
do
  pgrep -f $a > /dev/null ||  deadAgents[$(( ${#deadAgents} + 1 ))]=$a
done

# If there are no dead agents just finish
[[ ${#deadAgents} -eq 0 ]] && exit

logme $deadAgents where stopped and restarted now at $HOSTNAME.
echo $deadAgents where stopped and restarted now at $HOSTNAME. | mail -s "File management modules not Running" $EMAIL

if [[ ${#deadAgents} -eq ${#AGENTS} ]] 
then
  startAgents all
else 
  for a in $deadAgents
  do
    startAgents $a
  done
fi
