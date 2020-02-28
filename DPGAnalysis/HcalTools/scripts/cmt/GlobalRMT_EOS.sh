#!/bin/bash

WebDir='/eos/cms/store/group/dpg_hcal/comm_hcal/www/HcalRemoteMonitoring'
WebSite='https://cms-conddb-dev.cern.ch/eosweb/hcal/HcalRemoteMonitoring'
HistoDir='/store/group/dpg_hcal/comm_hcal/www/HcalRemoteMonitoring/CMT/histos'
eos='/afs/cern.ch/project/eos/installation/0.3.84-aquamarine/bin/eos.select'

# print usage info
if [[ "$1" == "" ]]; then
  echo "Usage:"
  echo "  $0 file [comment] [-ignore-file] [-das-cache]"
  echo "    file  -- a file with run numbers"
  echo "    comment  -- add a comment line (instead of spaces use '_')"
  echo "    -ignore-file   -- skips production of run html pages. Produces"
  echo "                      only the global page. File name is not needed."
  echo "    -das-cache   -- whether to save DAS information locally for a reuse"
  echo
  echo "example: ./GLOBAL.sh Run_List.txt"
  exit 1
fi

cmsenv 2>/dev/null
if [ $? == 0 ] ; then
    eval `scramv1 runtime -sh`
fi
temp_var=`ls ${eos}`
status="$?"
echo "using eos command <${temp_var}>"
if [ ! ${status} -eq 0 ] ; then
    echo "failed to find eos command"
    # exit 1
fi


# create log directory
LOG_DIR="dir-Logs"
if [ ! -d ${LOG_DIR} ] ; then mkdir ${LOG_DIR}; fi
rm -f ${LOG_DIR}/*


# Process arguments and set the flags
fileName=$1
comment=$2
if [ ${#comment} -gt 0 ] && [ "${comment:0:1}" == "-" ] ; then comment=""; fi
ignoreFile=0
debug=0
dasCache=0
DAS_DIR="d-DAS-info"

for a in $@ ; do
    if [ "$a" == "-ignore-file" ] ; then
	echo " ** file will be ignored"
	fileName=""
	ignoreFile=1
    elif [ "$a" == "-das-cache" ] ; then
	echo " ** DAS cache ${DAS_DIR} enabled"
	dasCache=1
	if [ ! -d ${DAS_DIR} ] ; then mkdir ${DAS_DIR}; fi
    else
	temp_var=${a/-debug/}
	if [ ${#a} -gt ${#temp_var} ] ; then
	    debug=${temp_var}
	    echo " ** debug detected (debug=${debug})"
	fi
    fi
done

# Obtain the runList from a file, if needed
runList=""
if [ ${#fileName} -gt 0 ] ; then
  if [ -s ${fileName} ] ; then
      runList=`cat ${fileName}`
  else
      echo "<${fileName}> does not seem to be a valid file"
      exit 2
  fi
else
    if [ ${ignoreFile} -eq 0 ] ; then
	echo " ! no file provided"
    fi
    echo " ! will produce only the global html page"
fi


# Check the runList and correct the correctables
# Replace ',' and ';' by empty spaces
runList=`echo "${runList}" | sed 'sk,k\ kg' | sed 'sk;k\ kg'`
ok=1
for r in ${runList} ; do
    if [ ! ${#r} -eq 6 ] ; then
	echo "run numbers are expected to be of length 6. Check <$r>"
	ok=0
    fi
    debug_loc=0
    if [ "$r" -eq "$r" ] 2>/dev/null ; then
	if [ ${debug_loc} -eq 1 ] ; then echo "run variable <$r> is a number (ok)"; fi
    else
	echo "error: run variable <$r> is not an integer number"
	ok=0
    fi
done

echo "Tested `wc -w <<< "${runList}"` runs from file ${fileName}"
if [ ${ok} -eq 0 ] ; then
    echo "errors in the file ${fileName} with run numbers"
    exit 3
else
    if [ ${#fileName} -gt 0 ] ; then
	echo "run numbers in ${fileName} verified ok"
    fi
fi

comment=`echo ${comment} | sed sk\_k\ kg`
if [ ${#comment} -gt 0 ] ; then
    echo "comment \"${comment}\" will be added to the pages"
fi

if [ ${debug} -eq 3 ] ; then exit; fi


echo 
echo 
echo 
echo 'Run numbers for processing'
echo "${runList}"
echo -e "list complete\n"

#processing

for i in ${runList} ; do
    runnumber=$i

    logFile="${LOG_DIR}/log_${runnumber}.out"
    rm -f ${logFile}

# if [[ "$runnumber" > 233890 ]] ; then
    echo 
    echo 
    echo
    echo  "Run for processing $runnumber"
    echo  "always copy root file from /eos !!!"
    echo  "file=root://eoscms//cms/$HistoDir/Global_$runnumber.root"
# always copy root file from /eos !!!
##    if [ ! -s Global_${runnumber}.root ] ; then
##	xrdcp root://eoscms//eos/cms/$HistoDir/Global_$runnumber.root Global_$runnumber.root
	xrdcp -f root://eoscms//eos/cms/$HistoDir/Global_$runnumber.root Global_$runnumber.root
	status="$?"
	if [ ! ${status} -eq 0 ] ; then
	    echo "failed to get file Global_${runnumber}.root"
	    exit 2
	fi
##    fi
    


    #GlobalRMT processing
    echo -e "\nRemoteMonitoringMAP_Global\n" >> ${logFile}
    ./RemoteMonitoringMAP_Global.cc.exe Global_$runnumber.root Global_$runnumber.root 2>&1 | tee -a ${logFile}
    if [ ! $? -eq 0 ] ; then
	echo "MAP_Global processing failed"
	exit 2
    fi

#    if [ ! -s HELP.html ] ; then
#	echo "MAP_Global failure was not detected. HELP.html is missing"
#	exit 2
#    fi

    local_WebDir=dir-GlobalRMT-GLOBAL_${runnumber}
    rm -rf ${local_WebDir}
    if [ ! -d ${local_WebDir} ] ; then mkdir ${local_WebDir}; fi
    for j in $(ls -r *.html); do
	cat $j | sed 's#cms-cpt-software.web.cern.ch\/cms-cpt-software\/General\/Validation\/SVSuite#cms-conddb-dev.cern.ch\/eosweb\/hcal#g' \
		> ${local_WebDir}/$j
    done

    cp *.png ${local_WebDir}
#    cp HELP.html ${local_WebDir}

    files=`cd ${local_WebDir}; ls`
    #echo "GlobalRMT files=${files}"

    if [ ${debug} -eq 0 ] ; then
	eos mkdir $WebDir/GlobalRMT/GLOBAL_$runnumber
	if [ ! $? -eq 0 ] ; then
	    echo "GlobalRMT eos mkdir failed"
	    exit 2
	fi
	for f in ${files} ; do
	    echo "eoscp ${local_WebDir}/${f} $WebDir/GlobalRMT/GLOBAL_$runnumber/${f}"
	    eoscp ${local_WebDir}/${f} $WebDir/GlobalRMT/GLOBAL_$runnumber/${f}
	    if [ ! $? -eq 0 ] ; then
		echo "GlobalRMT eoscp failed for ${f}"
		exit 2
	    fi
	done
    else
        # debuging
	echo "debugging: files are not copied to EOS"
    fi

    rm *.html
    rm *.png
    rm -rf dir-GlobalRMT-GLOBAL_*
    rm *.root
#fi

done

if [ ${debug} -eq 2 ] ; then
    echo "debug=2 skipping web page creation"
    exit 2
fi



echo "GlobalRMT script done"
