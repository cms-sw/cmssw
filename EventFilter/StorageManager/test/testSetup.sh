#
# 02-Jan-2009, KAB - This script sets up a bash environment
# for storage manager development.
# It needs to be run from the root directory of the SM development area.
#
# 22-Jan-2009, MO  - changes to make the script work from the
#                    EventFilter/StorageManager/test directory

export STMGR_DIR=`pwd`/demoSystem

# initial setup
source $STMGR_DIR/bin/uaf_setup.sh
#source $STMGR_DIR/bin/cvs_setup.sh

# check if this script is being run from inside a CMSSW project area
selectedProject=""
cmsswVersion=`pwd -P | sed -nre 's:.*/(CMSSW.*)/src/EventFilter/StorageManager/test.*:\1:p'`

if [[ "$cmsswVersion" != "" ]]
then
    selectedProject=`pwd -P | sed -nre "s:(.*/${cmsswVersion}).*:\1:p"`

else
    # check for existing project areas.  Prompt the user to choose one.
    projectCount=`ls -1d CMSSW* | wc -l`
    if [ $projectCount -eq 0 ]
    then
        echo "No project areas currently exist; try createProjectArea.csh"
        return
    fi
    projectList=`ls -dt CMSSW*`
    if [ $projectCount -eq 1 ]
    then
        selectedProject=`pwd -P`/$projectList
    else
        echo " "
        echo "Select a project:"
        for project in $projectList
        do
            echo -n "  Use $project (y/n [y])? "
            read response
            response=`echo ${response}y | tr "[A-Z]" "[a-z]" | cut -c1`
            if [ "$response" == "y" ]
            then
                selectedProject=`pwd -P`/$project
                break
            fi
        done
    fi
fi
if [ "$selectedProject" == "" ]
then
    echo "No project selected.  Exiting..."
    return
fi

# set up the selected project
cd ${selectedProject}/src
source $STMGR_DIR/bin/scram_setup.sh
cd - > /dev/null

scramArch=`scramv1 arch`
export PATH=${selectedProject}/test/${scramArch}:${PATH}

# define useful aliases

alias startEverything="cd $STMGR_DIR/bin; source ./startEverything.sh"

alias startConsumer="cd $STMGR_DIR/log/client; cmsRun ../../cfg/eventConsumer.py"
alias startConsumer1="cd $STMGR_DIR/log/client1; cmsRun ../../cfg/eventConsumer.py"
alias startConsumer2="cd $STMGR_DIR/log/client2; cmsRun ../../cfg/eventConsumer.py"

alias startProxyConsumer="cd $STMGR_DIR/log/client1; cmsRun ../../cfg/proxyEventConsumer.py"

alias startProxyDQMConsumer="cd $STMGR_DIR/log/client; cmsRun ../../cfg/proxyDQMConsumer.py"

alias startDQMConsumer="cd $STMGR_DIR/log/client; cmsRun ../../cfg/dqmConsumer.py"

alias cleanupShm="FUShmCleanUp_t"
alias killEverything="killall -9 xdaq.exe; sleep 2; FUShmCleanUp_t; cd $STMGR_DIR/bin; ./removeOldLogFiles.sh; ./removeOldDataFiles.sh; ./removeOldDQMFiles.sh; cd - > /dev/null"

alias globalConfigure="cd $STMGR_DIR/soap; ./globalConfigure.sh"
alias globalEnable="cd $STMGR_DIR/soap; ./globalEnable.sh"
alias globalStop="cd $STMGR_DIR/soap; ./globalStop.sh"
alias globalHalt="cd $STMGR_DIR/soap; ./globalHalt.sh"

alias shutdownEverything="globalStop ; sleep 3 ; killEverything"

# 02-Jan-2008 - if needed, create a shared memory key file so that we
# can use shared memory keys independent of other developers
keyDir="/tmp/$USER"
if [ ! -d $keyDir ]
then
    mkdir $keyDir
fi

keyFile="$keyDir/shmKeyFile"
touch $keyFile
export FUSHM_KEYFILE=$keyFile
export SMPB_SHM_KEYFILE=$keyFile

keyFile="$keyDir/semKeyFile"
touch $keyFile
export FUSEM_KEYFILE=$keyFile

# 02-Jan-2009 - define the number of FUs that we want
# Valid values are 1..8.
export SMDEV_FU_PROCESS_COUNT=2

# 02-Jan-2009 - define whether we want a big HLT config or not
# Valid values are 0 (small config) and 1 (big config)
export SMDEV_BIG_HLT_CONFIG=0

# 08-JUL-2009 - define the configuration to be used
#export STMGR_CONFIG=$STMGR_DIR/cfg/sm_autobu_8fu.xml
export STMGR_CONFIG=$STMGR_DIR/cfg/sm_autobu_8fu_atcp.xml
