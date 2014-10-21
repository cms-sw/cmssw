#! /bin/bash

#export WorkDir=$(dirname $0)
YourEmail=sdutta@cern.ch
#source /nfshome0/cmssw2/scripts/setup.sh

export SCRAM_ARCH=slc5_amd64_gcc462
if [[ ! $HOME =~ /nfshome0/dqm* ]]
then 
  HOME=/nfshome0/${USER/local/}
fi
if [[ -d ${HOME}/prod || -d ${HOME}/dev ]] 
then
  source /nfshome0/dqmpro/bin/setup_cmssw.sh
  [[ -d ${HOME}/prod ]] && cd ${HOME}/prod || cd ${HOME}/dev
  eval `scram runtime -sh`
else
  source $WorkDir/env3.sh
fi
export PYTHONPATH=$XPYTHONPATH:$PYTHONPATH
export HOSTNAME=$HOSTNAME
agents_pnames=("fileCollector" "producerFileCleanner")
agents_executables=("/nfshome0/dqmpro/filecollector/fileCollector2.py" "/nfshome0/dqmpro/filecollector/producerFileCleanner.py")
if [[ $USER =~ 'dqmpr.*' ]]
then
  agents_parameters=("/home/dqmprolocal/output /home/dqmprolocal/done /dqmdata/dqm/uploads" \
                     "/cmsnfshome0/nfshome0/dqmpro/filecollector/RootArchivalAndTransferSystem_cfg.py")
else
  agents_parameters=("/home/dqmdevlocal/output /home/dqmdevlocal/done /dqmdata/dqmintegration/upload" \
                     "/cmsnfshome0/nfshome0/dqmpro/filecollector/RootArchivalAndTransferSystem_cfg.py")
fi
WorkDir=$( dirname ${agents_executables[0]} )
[[ -e $WorkDir/.start ]] && [[ -e $WorkDir/.stop ]] && rm $WorkDir/.stop
[[ -e $WorkDir/.stop ]] && echo Found stop file not starting the agents && exit 0

msg=
new_line=
for pos in $(seq 0 $(( ${#agents_executables[@]} - 1 ))); do
  RUN_STAT=`ps -ef | grep -P "(${agents_executables[$pos]})" | grep -v grep | wc | awk '{print $1}'`
  if [ $RUN_STAT -ne 0 ];then
    echo ${agents_pnames[$pos]} is running
  else
    echo ${agents_pnames[$pos]} stopped by unknown reason and restarted now.
    TIMETAG=$(date +"%Y%m%d_%H%M%S")
    LOG=$WorkDir/log/LOG.${agents_pnames[$pos]}.$HOSTNAME.$TIMETAG
    ${agents_executables[$pos]} ${agents_parameters[$pos]} >& $LOG &
    date >> $LOG
    [[ ! -e $WorkDir/.start ]] && 
         echo ${agents_pnames[$pos]} stopped by unknown reason and restarted at $HOSTNAME. >> $LOG ||
         echo ${agents_pnames[$pos]} Found .start file, starting
    [[ ! -z $msg ]] && new_line="\n"    
    msg=$msg$new_line${agents_pnames[$pos]}" stopped by unknown reason and restarted now at $HOSTNAME."
  fi
done

[[ ! -e $WorkDir/.start && ! -z $msg ]] && echo $msg | mail -s "File Collection Agents not Running" $YourEmail

if [[ -e $WorkDir/.start ]]
then
  sleep 10
  master=$(cat $WorkDir/.start)
  [[ $(hostname -s) == $master ]] && rm $WorkDir/.start
fi 
