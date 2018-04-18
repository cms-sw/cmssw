#!/bin/sh
#
# set the CMSSW environtmet for SiStripO2O
# and the proxies for conditionUploader
#

O2ONAME=$1

# deployment-specific params
BASEDIR=/data/O2O
SCRAM_ARCH=slc6_amd64_gcc630
RELEASE=CMSSW_10_0_5
RELEASEDIR=/cvmfs/cms.cern.ch/slc6_amd64_gcc630/cms/cmssw/${RELEASE}

source /cvmfs/cms.cern.ch/cmsset_default.sh
cd ${RELEASEDIR}/src
eval `scramv1 ru -sh`
cd -

O2O_HOME=/data/O2O/SiStrip/

# for sqlalchmey (?)
export PYTHON_EGG_CACHE=/data/condbpro
# path to .netrc file and .cms_cond dir
export COND_AUTH_PATH=/data/O2O/SiStrip

# save log files produced by o2oRun.py on disk
export JOBDIR=${O2O_HOME}/jobs/${O2ONAME}
export O2O_LOG_FOLDER=${BASEDIR}/logs/${O2ONAME}
export LOGFILE=${BASEDIR}/logs/$O2ONAME.log

# temperoray fix for TNS_ADMIN
export TNS_ADMIN=/cvmfs/cms.cern.ch/slc6_amd64_gcc530/cms/oracle-env/29/etc
