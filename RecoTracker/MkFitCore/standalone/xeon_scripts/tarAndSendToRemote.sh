#! /bin/bash

########################
## Command Line Input ##
########################

remote_arch=${1} # SNB, KNL, SKL-SP
suite=${2:-"forPR"} # which set of benchmarks to run: full, forPR, forConf
useARCH=${3:-0}
lnxuser=${4:-${USER}}

###################
## Configuration ##
###################

source xeon_scripts/common-variables.sh ${suite} ${useARCH} ${lnxuser}
source xeon_scripts/init-env.sh

# architecture dependent settings
if [[ "${remote_arch}" == "SNB" ]]
then
    HOST=${SNB_HOST}
    DIR=${SNB_WORKDIR}/${SNB_TEMPDIR}
elif [[ "${remote_arch}" == "KNL" ]]
then
    HOST=${KNL_HOST}
    DIR=${KNL_WORKDIR}/${KNL_TEMPDIR}
elif [[ "${remote_arch}" == "LNX-G" ]]
then
    HOST=${LNXG_HOST}
    DIR=${LNXG_WORKDIR}/${LNXG_TEMPDIR}
elif [[ "${remote_arch}" == "LNX-S" ]]
then
    HOST=${LNXS_HOST}
    DIR=${LNXS_WORKDIR}/${LNXS_TEMPDIR}
else 
    echo ${remote_arch} "is not a valid architecture! Exiting..."
    exit
fi

##################
## Tar and Send ##
##################

assert_settings=true
echo "--------Showing System Settings--------"
# unzip tarball remotely
echo "Untarring repo on ${remote_arch} remotely"
SSHO ${HOST} bash -c "'
echo "--------Showing System Settings--------"
##### Check Settings #####
echo "turbo status: "$(cat /sys/devices/system/cpu/intel_pstate/no_turbo)
echo "scaling governor setting: "$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)
echo "--------End System Settings ------------"
if ${assert_settings};
then
echo "Ensuring correct settings"
if [[ $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor) != "performance" ]]
then
echo "performance mode is OFF. Exiting"
exit 1
fi
if [[ $(cat /sys/devices/system/cpu/intel_pstate/no_turbo) == "0" ]]
then
echo "Turbo is ON. Exiting"
exit 1
fi
fi
sleep 3 ## so you can see the settings
'"
bad=$(SSHO ${HOST} echo $?)
if [ $bad -eq 1 ]; then
echo "killed"
exit 1
fi

# tar up the directory
echo "Tarring directory for ${remote_arch}... make sure it is clean!"
repo=mictest.tar.gz
tar --exclude-vcs --exclude='*.gz' --exclude='validation*' --exclude='*.root' --exclude='log_*' --exclude='*.png' --exclude='*.o' --exclude='*.om' --exclude='*.d' --exclude='*.optrpt' -zcvf  ${repo} *

# mkdir tmp dir on remote arch
echo "Making tmp dir on ${remote_arch} remotely"
SSHO ${HOST} bash -c "'
mkdir -p ${DIR}
exit
'"

# copy tarball
echo "Copying tarball to ${remote_arch}"
scp ${repo} ${HOST}:${DIR}

# unzip tarball remotely
echo "Untarring repo on ${remote_arch} remotely"
SSHO ${HOST} bash -c "'
cd ${DIR}
tar -zxvf ${repo}
rm ${repo}
'"

# remove local tarball
echo "Remove local repo tarball"
rm ${repo}
