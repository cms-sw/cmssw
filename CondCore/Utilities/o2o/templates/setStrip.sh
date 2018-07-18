#!/bin/sh
#
# set the CMSSW environtmet for SiStripO2O
# and the proxies for conditionUploader
#

O2ONAME=$1

# deployment-specific params
BASEDIR=@root
SCRAM_ARCH=@arch
RELEASE=@release
RELEASEDIR=@cmsswroot/@arch/cms/cmssw/${RELEASE}

source @cmsswroot/cmsset_default.sh
cd ${RELEASEDIR}/src
eval `scramv1 ru -sh`
cd -

O2O_HOME=$BASEDIR/SiStrip/

# for sqlalchmey (?)
export PYTHON_EGG_CACHE=@localhome
# path to .netrc file and .cms_cond dir
export COND_AUTH_PATH=@root/SiStrip

# save log files produced by o2oRun.py on disk
export JOBDIR=${O2O_HOME}/jobs/${O2ONAME}
export O2O_LOG_FOLDER=${BASEDIR}/logs/${O2ONAME}
export LOGFILE=${BASEDIR}/logs/$O2ONAME.log

# temperoray fix for TNS_ADMIN
export TNS_ADMIN=@extroot/oracle-env/29/etc
