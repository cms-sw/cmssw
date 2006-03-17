#! /bin/sh
########################################################################
#Prerequesits:
# --you have an application which writes offline data, example see CMSSW/src/CondTools/Ecal/test/OfflinePedWriter.cpp or CMSSW/src/CondCore/ESSources/test/testWriteCalib.cpp
# --you have bootstraped a CMSSW scram project from where your writer application runs
#INSTRUCTION:
# --download this script and change the section "user defined parameters" and the section "user defined application" according to your setup 
# --run this script in the directory just outside your working CMSSW project
################setting up TNS_ADMIN######################################
#
#uncomment this if your working node doesnot recognise devdb10 service
#
#TNSFILE=tnsnames.ora
#rm -f ${TNSFILE}
#/bin/cat >  ${TNSFILE} <<EOI
#devdb10=(DESCRIPTION=
#        (ADDRESS=
#                (PROTOCOL=TCP)
#                (HOST=oradev10.cern.ch)
#                (PORT=10520)
#        )
#        (CONNECT_DATA=
#                (SID=D10))
#        )
#EOI
#
#export TNS_ADMIN=${THISDIR}
##################user defined parameters#################################
THISDIR=`pwd`
export SCRAM_ARCH=slc3_ia32_gcc323
CMSSWVERSION=CMSSW_2006-02-24
OWNER=ECAL
##################set up catalog and connection############################
cd ${CMSSWVERSION}/src
eval `scramv1 runtime -sh`
cd ${THISDIR}
export CORAL_AUTH_USER=CMS_COND_${OWNER}
export CORAL_AUTH_PASSWORD=cern2006x #change to your password!!
rm -f conddbcatalog.xml
echo "Publishing catalog"
echo "Using TNS_ADMIN ${TNS_ADMIN}"
FCpublish -u relationalcatalog_oracle://devdb10/CMS_COND_GENERAL -d file:conddbcatalog.xml
#OWNERPASS=`echo ${OWNER} | gawk '{print substr(tolower($1),0,3);}'`
export POOL_CATALOG=file:${THISDIR}/conddbcatalog.xml
####################user defined application##############################
###run your writer application here, in the same script and change whatever parameters are required by your application
##########################################################################
MYAPP=OfflinePedWriter  #your writer application name
echo "Running ${MYAPP}"
export CONNECT=oracle://devdb10/${CORAL_AUTH_USER}
export TAG=ecal_test      #tag. please change to your tag!!!
echo "Using TNS_ADMIN ${TNS_ADMIN}"
${LOCALRT}/test/${SCRAM_ARCH}/${MYAPP} ${CONNECT} 10 ${TAG}
