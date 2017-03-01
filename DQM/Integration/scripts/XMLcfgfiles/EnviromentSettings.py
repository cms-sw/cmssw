def detectEnviroment():
	global environmentString,CMSSW_Release,CMSSW_Release_Number
	import sys,os
	envVariables={"CMSSW_BASE":"","CMSSW_DATA_PATH":"","CMSSW_RELEASE_BASE":"","CMSSW_SEARCH_PATH":"",
					"CMSSW_VERSION":CMSSW_Release,"LD_LIBRARY_PATH":"","POOL_OUTMSG_LEVEL":"","POOL_STORAGESVC_DB_AGE_LIMIT":"",
					"PYTHONPATH":"","PATH":"","ROOTSYS":"","SEAL":"","SEAL_KEEP_MODULES":"","SEAL_PLUGINS":"","TNS_ADMIN":"",
					"XDAQ_DOCUMENT_ROOT":"","XDAQ_OS":"","XDAQ_PLATFORM":"","XDAQ_ROOT":"","XDAQ_SETUP_ROOT":"",
					"XDAQ_ZONE":"","SCRAM_ARCH":""}
	cmd='env | grep  -E "(%s)" | sort' % "|".join(["^%s=" % v for v in envVariables.keys()])
	fpd=os.popen(cmd)
	line=fpd.readline()
	while line:
		k,v=line.split("=")
		envVariables[k]=v.strip()
		line=fpd.readline()
	fpd.close()
	ppath=":".join(sys.path)
	ppath=ppath.strip(":")
	envVariables["PYTHONPATH"]=ppath
	envVariables["PATH"]="%s:${PATH}" % envVariables["PATH"]
	envVariables["XDAQ_SETUP_ROOT"]=envVariables["XDAQ_ROOT"]+"/share"
	envVariables["XDAQ_ZONE"]="cdaq"
	CMSSW_Release=envVariables["CMSSW_VERSION"]
	CMSSW_Release_Number=CMSSW_Release.split("_",1)[1]
	#envVariables["LD_LIBRARY_PATH"]=envVariables["XDAQ_ROOT"]+"/lib:/nfshome0/dqmpro/lib_"+CMSSW_Release_Number
	for (key,value) in envVariables.items():
		envVariables[key]=value.replace("/cmsnfshome0","")
	return " ".join(["%s=%s" % (k,v) for (k,v) in envVariables.items()])
################################################################################
user=""
users=["dqmpro","dqmdev"]
userHomeDirecotry=lambda: "/nfshome0/"+user
#Default values however they get updateted to the actual release after the call 
#to detectEnviroment()
CMSSW_Release_Number="2_2_10"
CMSSW_Release="CMSSW_"+CMSSW_Release_Number

environmentString=detectEnviroment()

libDirectory=lambda: userHomeDirecotry()+"/lib_"+CMSSW_Release_Number
knownclassNames=lambda: {
	"xmas2dqm::wse::XmasToDQM"	:{"urn":"/urn:xdaq-application:lid=52","modulePath":libDirectory()+"/libDQMServicesXdaqCollector.so","tid":["104"]},
	"evf::FUEventProcessor"		:{"urn":"/urn:xdaq-application:lid=50","modulePath":"lib/libxdaq2rc.so "+libDirectory()+"/libEventFilterProcessor.so","tid":["100","102"]},
	"SMProxyServer"				:{"urn":"/urn:xdaq-application:lid=30","modulePath":"lib/libxdaq2rc.so "+libDirectory()+"/libEventFilterSMProxyServer.so","tid":["102"]},
	"RCMSStateListener"			:{"urn":"","modulePath":"","tid":["104"]}
}

#Function manager parameters
host={	"dqmpro":"cmsrc-dqm",
		"dqmdev":"cmsrc-dqmdev"}
port={	"dqmpro":"22000",
		"dqmdev":"42000"}
sourceURL={	"dqmpro":"http://cmsrc-dqm:22000/functionmanagers/dqmFM.jar",
			"dqmdev":"http://cmsrc-dqmdev:42000/functionmanagers/dqmFM.jar"}
			
RCMSStateListenerURL={	"dqmpro":"http://cmsrc-dqm:22001/rcms/services/replycommandreceiver",
						"dqmdev":"http://cmsrc-dqmdev.cms:42001/rcms/services/replycommandreceiver"}


#XdaqExecutive parameters

pathsToExecutive={	"dqmpro":"/opt/xdaq/bin/xdaq.sh -e /nfshome0/dqmpro/xml/profile.xml -u xml://cmsrc-dqm:22010",
					"dqmdev":"/opt/xdaq/bin/xdaq.sh -e /nfshome0/dqmdev/xml/profile.xml -u xml://cmsrc-dqmdev:42010"}
logURLs= {	"dqmpro":"xml://cmsrc-dqm:22010",
			"dqmdev":"xml://cmsrc-dqmdev:42010"}
knownlogLevels=["DEBUG","INFO","ERROR"]
