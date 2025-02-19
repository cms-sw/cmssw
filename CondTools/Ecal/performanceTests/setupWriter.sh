#! /bin/bash

source setup.sh

########################################################################
#Prerequesits:
# --you have an application which writes offline data, example see CMSSW/src/CondTools/Ecal/test/OfflinePedWriter.cpp or CMSSW/src/CondCore/ESSources/test/testWriteCalib.cpp
# --you have bootstraped a CMSSW scram project from where your writer application runs
#INSTRUCTION:
# --download this script and change the section "user defined parameters" and the section "user defined application" according to your setup 
# --run this script in the directory just outside your working CMSSW project

##################user defined parameters#################################
THISDIR=`pwd`
CMSSWVERSION=CMSSW_0_2_0_pre7 #change to the CMSSW version for testing
export CVSROOT=:kserver:cmscvs.cern.ch:/cvs_server/repositories/CMSSW
export SCRAM_ARCH=slc3_ia32_gcc323
SERVICENAME=cms_val_lb
SERVERNAME=${SERVICENAME}.cern.ch
SERVERHOSTNAME=int2r1-v.cern.ch
GENREADER=CMS_VAL_GENERAL_POOL_READER
GENREADERPASS=val_gen_rea_1031
GENSCHEMA=CMS_VAL_GENERAL_POOL_OWNER
OWNER=ECAL
MYAPP=OfflinePedWriter  #your writer application name
##################set up catalog and connection############################
bootstrap_cmssw $CMSSWVERSION
setup_tns
export POOL_AUTH_USER=CMS_VAL_GENERAL_POOL_OWNER
export POOL_AUTH_PASSWORD=val_gen_own_1031
rm -f conddbcatalog.xml
echo "Publishing catalog"
echo "Using TNS_ADMIN ${TNS_ADMIN}"
FCpublish -u relationalcatalog_oracle://cms_val_lb.cern.ch/CMS_VAL_GENERAL_POOL_OWNER -d file:conddbcatalog.xml
OWNERPASS=`echo ${OWNER} | gawk '{print substr(tolower($1),0,3);}'`
export POOL_AUTH_USER=CMS_VAL_${OWNER}_POOL_OWNER
export POOL_AUTH_PASSWORD=val_${OWNERPASS}_own_1031
export POOL_CATALOG=file:${THISDIR}/conddbcatalog.xml
####################user defined application##############################
###run your writer application here, in the same script and change whatever parameters are required by your application
##########################################################################
echo "Running ${MYAPP}"
export CONNECT=oracle://cms_val_lb.cern.ch/${POOL_AUTH_USER}
export TAG=ecalfall_test      #tag
echo "Using TNS_ADMIN ${TNS_ADMIN}"
${CMSSWVERSION}/test/${SCRAM_ARCH}/${MYAPP} ${CONNECT} 10000 ${TAG}
