#!/bin/bash

# setup scripts for read/write DB tests

# ------------------------------------------------------------------------
# setup_tns ()
# write tnsnames.ora in the current directory and set variable TNS_ADMIN for running this job to this directory
# Parameters:
# Returns; 0 on success
# ------------------------------------------------------------------------
setup_tns() {
local TNSFILE=tnsnames.ora
rm -f ${TNSFILE}
/bin/cat >  ${TNSFILE} <<EOI
${SERVERNAME} =
   (DESCRIPTION =
     (ADDRESS = (PROTOCOL = TCP)(HOST = ${SERVERHOSTNAME})(PORT = 1521))
     (LOAD_BALANCE = yes)
     (CONNECT_DATA =
       (SERVER = DEDICATED)
       (SERVICE_NAME = ${SERVICENAME})
     )
)
EOI
export TNS_ADMIN=${THISDIR}
echo "[---JOB LOG---] Using TNS_ADMIN=${TNS_ADMIN}, ORACLE server ${SERVERNAME} at host ${SERVERHOSTNAME}"
return 0
}
#-------------------------------------------------------------------------
# bootstrap_cmssw ()
# bootstrap a new CMSSW project, removing old if exists
# Parameters: CMSSWVERSION($1)
# Returns; 0 on success
#-------------------------------------------------------------------------
bootstrap_cmssw () {
if [ -d $1 ]; 
    #then /bin/rm -rf $1 ;
    then 
    echo "[---JOB LOG---] $1 already exists, using that"
    cd $1/src
    eval `scramv1 runtime -sh`
    cd ${THISDIR}
    return 0
fi 
scramv1 project CMSSW $1
cd $1/src
eval `scramv1 runtime -sh`
cvs co CondCore/PluginSystem
cvs co CondCore/MetaDataService
cvs co CondTools/Ecal
scramv1 b
cd ${THISDIR}
return 0
}
# ------------------------------------------------------------------------
# write_config ()
# write the job configuration file
# Parameters: OWNER($1), OBJNAME($2), MAXEVENTS($3), FIRSTRUN($4), EVENTSINRUN($5)
# Returns; 0 on success
# ------------------------------------------------------------------------
write_config() {
CONFFILE=condRead$2.cfg
local CONNECT=oracle://${SERVERNAME}/CMS_VAL_${1}_POOL_OWNER
local RCD=$2Rcd
local TAG=`echo $1| awk '{print tolower($1)}'`fall_test
/bin/cat >  ${CONFFILE} <<EOI
  process condTEST = {
	path p = { get & print }

	es_source = PoolDBESSource { VPSet toGet = {
                                   {string record = "${RCD}"
                                     string tag = "${TAG}"
                                    } }
		    		    bool loadAll = true
                                    string connect = "${CONNECT}"
			            string timetype = "runnumber" 
				   }

	module print = AsciiOutputModule { }
	
	source = EmptySource {untracked int32 maxEvents = $3 
                untracked uint32 firstRun = $4 
                untracked uint32 numberEventsInRun = $5}

	module get = EventSetupRecordDataGetter { VPSet toGet = {
	       {string record = "${RCD}"
	        vstring data = {"$2"} } 
	       }
	       untracked bool verbose = true 
	}
}
EOI
return 0
}
