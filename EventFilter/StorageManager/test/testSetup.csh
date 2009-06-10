#
# 15-Nov-2007, KAB - This script sets up the environment
# for storage manager development.
# It needs to be run from the root directory of the SM development area.
#
# 02-Jan-2009, KAB - cleanup before CVS checkin
# 22-Jan-2009, MO  - changes to make the script work from the
#                    EventFilter/StorageManager/test directory

set demoSystemDir = `pwd`/demoSystem
setenv STMGR_DIR $demoSystemDir

# initial setup
source $demoSystemDir/bin/uaf_setup.csh
#source $demoSystemDir/bin/cvs_setup.csh

# check if this script is being run from inside a CMSSW project area
set selectedProject = ""
set cmsswVersion = `pwd | sed -nre 's:.*/(CMSSW.*)/src/EventFilter/StorageManager/test.*:\1:p'`

if ($cmsswVersion != "") then
    set selectedProject = `pwd | sed -nre "s:(.*/${cmsswVersion}).*:\1:p"`

else
    # check for existing project areas.  Prompt the user to choose one.
    set projectCount = `ls -1d CMSSW* | wc -l`
    if ($projectCount == 0) then
        echo "No project areas currently exist; try createProjectArea.csh"
        exit
    endif
    set projectList = `ls -dt CMSSW*`
    if ($projectCount == 1) then
        set selectedProject = `pwd`/$projectList
    else
        echo " "
        echo "Select a project:"
        foreach project ($projectList)
            echo -n "  Use $project (y/n [y])? "
            set response = $<
            set response = `echo ${response}y | tr "[A-Z]" "[a-z]" | cut -c1`
            if ($response == "y") then
                set selectedProject = `pwd`/$project
                break
            endif
        end
    endif
endif
if ($selectedProject == "") then
    echo "No project selected.  Exiting..."
    exit
endif

# set up the selected project
cd ${selectedProject}/src
source $demoSystemDir/bin/scram_setup.csh
cd -

set scramArch = `scramv1 arch`
setenv PATH ${selectedProject}/test/${scramArch}:${PATH}

# define useful aliases

alias startEverything "cd $demoSystemDir/bin; source ./startEverything.csh"

# this test may need to be more sophisticated if Python configs
# are enabled in 2_2_X...
if ($selectedProject =~ *CMSSW_3_*) then
    alias startConsumer "cd $demoSystemDir/log/client; cmsRun ../../cfg/eventConsumer.py"
    alias startConsumer1 "cd $demoSystemDir/log/client1; cmsRun ../../cfg/eventConsumer.py"
    alias startConsumer2 "cd $demoSystemDir/log/client2; cmsRun ../../cfg/eventConsumer.py"

    alias startProxyConsumer "cd $demoSystemDir/log/client1; cmsRun ../../cfg/proxyEventConsumer.py"

    alias startProxyDQMConsumer "cd $demoSystemDir/log/client; cmsRun ../../cfg/proxyDQMConsumer.py"

    alias startDQMConsumer "cd $demoSystemDir/log/client; cmsRun ../../cfg/dqmConsumer.py"
else
    alias startConsumer "cd $demoSystemDir/log/client; cmsRun ../../cfg/eventConsumer.cfg"
    alias startConsumer1 "cd $demoSystemDir/log/client1; cmsRun ../../cfg/eventConsumer.cfg"
    alias startConsumer2 "cd $demoSystemDir/log/client2; cmsRun ../../cfg/eventConsumer.cfg"

    alias startProxyConsumer "cd $demoSystemDir/log/client1; cmsRun ../../cfg/proxyEventConsumer.cfg"

    alias startProxyDQMConsumer "cd $demoSystemDir/log/client; cmsRun ../../cfg/proxyDQMConsumer.cfg"

    alias startDQMConsumer "cd $demoSystemDir/log/client; cmsRun ../../cfg/dqmConsumer.cfg"
endif

alias cleanupShm "FUShmCleanUp_t"
alias killEverything "killall -9 xdaq.exe; sleep 2; FUShmCleanUp_t; cd $demoSystemDir/bin; ./removeOldLogFiles.sh; ./removeOldDataFiles.sh; ./removeOldDQMFiles.sh; cd -"

alias globalConfigure "cd $demoSystemDir/soap; ./globalConfigure.csh"
alias globalEnable "cd $demoSystemDir/soap; ./globalEnable.csh"
alias globalStop "cd $demoSystemDir/soap; ./globalStop.csh"
alias globalHalt "cd $demoSystemDir/soap; ./globalHalt.csh"

alias shutdownEverything "globalStop ; sleep 3 ; killEverything"

# 02-Jan-2008 - if needed, create a shared memory key file so that we
# can use shared memory keys independent of other developers
set keyDir = "/tmp/$USER"
if (! (-d $keyDir)) then
    mkdir $keyDir
endif

set keyFile = "$keyDir/shmKeyFile"
touch $keyFile
setenv FUSHM_KEYFILE $keyFile
setenv SMPB_SHM_KEYFILE $keyFile

set keyFile = "$keyDir/semKeyFile"
touch $keyFile
setenv FUSEM_KEYFILE $keyFile

# 02-Jan-2009 - define the number of FUs that we want
# Valid values are 1..8.
setenv SMDEV_FU_PROCESS_COUNT 2

# 02-Jan-2009 - define whether we want a big HLT config or not
# Valid values are 0 (small config) and 1 (big config)
setenv SMDEV_BIG_HLT_CONFIG 0
