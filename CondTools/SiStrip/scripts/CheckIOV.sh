#!/bin/sh



function getLastIOV(){
    tag=$1
#    lastIOV=`cmscond_list_iov -c oracle://devdb10/CMS_COND_STRIP -u CMS_COND_STRIP -p w3807dev -f "relationalcatalog_oracle://devdb10/CMS_COND_GENERAL" -t $1 | grep DB | awk '{print $1}' | tail -1`
    lastIOV=`cmscond_list_iov -c oracle://orcon/CMS_COND_STRIP -u CMS_COND_STRIP_R -p R2106xon -f "relationalcatalog_oracle://orcon/CMS_COND_GENERAL" -t $1 | grep DB | awk '{print $1}' | tail -1`
    echo $lastIOV
}

function getFromRunSummaryTIF(){

    wget -q -r "http://cmsdaq.cern.ch/cmsmon/cmsdb/servlet/RunSummaryTIF?RUN_BEGIN=$1&RUN_END=$1&TEXT=1&DB=omds" -O out.txt

    cat out.txt | awk -F"\t" '{if (NR>1) print $11}' 
}

############
##  MAIN  ##
############

function CheckIOV(){

    if [ "$#" != 2 ]; then
	    echo -e "\n[usage] CheckIOV <RunNumber> <tag> "
	    return 4
    fi

    Run=$1
    tagPN=""
    [ "$2" != "" ] && tagPN=$2
    
    #//Verify that FEDVersion is not NULL
    FEDVersion_Run=`getFromRunSummaryTIF $Run`
    #echo ${FEDVersion_Run}
    [ "${FEDVersion_Run}" == "" ] && return 2  #//FedVersion NULL: perhaps you are asking for a not existing runNumber 

    #//Get lastIOV from OfflineDB//
    lastIOV=`getLastIOV $tagPN`
    #echo $lastIOV
    if [ "$lastIOV" == "" ]; then
    #//tag $tagPN not found in orcon, check failed//
	return 1
    fi
    if [ "$lastIOV" -ge "$Run" ]; then
    #//tag $tagPN found in orcon, Run inside a closed IOV, check successful//
	return 0
    fi
    
    #//Check FEDVersion of $Run and $lastIOV//
    FEDVersion_lastIOV=`getFromRunSummaryTIF $lastIOV`
    #echo ${FEDVersion_lastIOV}
    [ "${FEDVersion_lastIOV}" == "${FEDVersion_Run}" ] && return 0
    return 3
}
