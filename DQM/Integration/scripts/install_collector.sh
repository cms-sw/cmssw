
#!/bin/bash

source ~cmssw/cmsset_default.sh
CMSSW_VERSION=CMSSW_1_8_0
DQM_BASE=/home/dqm                  # Choose a directory of your liking
mkdir -p $DQM_BASE/collector

# setup environment
cd $DQM_BASE
scramv1 p CMSSW $CMSSW_VERSION

eval `cd $DQM_BASE/$CMSSW_VERSION && scramv1 run -sh`

# remove files
if [ -f $DQM_BASE/collector/collector-restart.sh ] 
then
 rm $DQM_BASE/collector/collector-restart.sh 
fi

if [ -f $DQM_BASE/collector/collector.crontab ]
then
 rm $DQM_BASE/collector/collector.crontab 
fi

# create files
export > $DQM_BASE/collector/collector-restart.sh

cat >> $DQM_BASE/collector/collector-restart.sh  << EOF

killall DQMCollector
if [ -f $DQM_BASE/collector/DQMCollector.out ] 
then 
 mv $DQM_BASE/collector/DQMCollector.out $DQM_BASE/collector/DQMCollector.\`date "+%y%m%d%H%M%S"\`
fi

nohup DQMCollector > $DQM_BASE/collector/DQMCollector.out &

EOF
chmod a+x $DQM_BASE/collector/collector-restart.sh

cat > $DQM_BASE/collector/collector.crontab << EOF
0,15,30,45 * * * 0-6 $DQM_BASE/collector/collector-restart.sh
EOF

# start collector either by hand or by crontab

