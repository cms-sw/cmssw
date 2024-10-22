#! /bin/bash

# command line input
dir=${1:-"benchmarks"} # Main output dir name
suite=${2:-"forPR"} # which set of benchmarks to run: full, forPR, forConf
afs_or_eos=${3:-"eos"} # which user space to use: afs or eos
lxpuser=${4:=${USER}}

# in case this is run alone
source xeon_scripts/common-variables.sh ${suite}
source xeon_scripts/init-env.sh

# first tar the directory to be sent
echo "Tarring plot directory"
tarball=${dir}.tar.gz
tar -zcvf ${tarball} ${dir}

# vars for LXPLUS
LXPLUS_HOST=${lxpuser}@lxplus.cern.ch
LXPLUS_OUTDIR=www
LXPLUS_WORKDIR=user/${lxpuser:0:1}/${lxpuser}

if [[ "${afs_or_eos}" == "afs" ]]
then
    LXPLUS_WORKDIR=/afs/cern.ch/${LXPLUS_WORKDIR}
elif [[ "${afs_or_eos}" == "eos" ]]
then
    LXPLUS_WORKDIR=/eos/${LXPLUS_WORKDIR}
else
    echo "${afs_or_eos} is not a valid option! Choose either 'afs' or 'eos'! Exiting..."
    exit
fi

# then send it!
scp -r ${tarball} ${LXPLUS_HOST}:${LXPLUS_WORKDIR}/${LXPLUS_OUTDIR}

# Make outdir nice and pretty
if [[ "${afs_or_eos}" == "afs" ]]
then
    echo "Unpacking tarball and executing remotely: ./makereadable.sh ${dir}"
    SSHO ${LXPLUS_HOST} bash -c "'
    cd ${LXPLUS_WORKDIR}/${LXPLUS_OUTDIR}
    tar -zxvf ${tarball}
    ./makereadable.sh ${dir}
    rm -rf ${tarball}
    exit
    '"
else
    echo "Unpacking tarball"
    SSHO ${LXPLUS_HOST} bash -c "'
    cd ${LXPLUS_WORKDIR}/${LXPLUS_OUTDIR}
    tar -zxvf ${tarball}
    rm -rf ${tarball}
    exit
    '"
fi

# remove local tarball
echo "Removing local tarball of plots"
rm ${tarball}

# Final message
echo "Finished tarring and sending plots to LXPLUS!"
