#!/bin/bash

# needed to allow the loop on *.png without using "*.png" as value
shopt -s nullglob
date

if [ $# -ne 5 ]; then
    afstokenchecker.sh "You have to provide a <DB>, an <Account>, a <GTAccount>, a <RunInfoAccount> and the <FrontierPath> !!!"
    exit
fi

afstokenchecker.sh "Starting execution of Monitor_RunInfo $1 $2 $3 $4 $5"

#Example: DB=cms_orcoff_prod
DB=$1
#Example: ACCOUNT=CMS_COND_21X_STRIP
ACCOUNT=$2
#Example: GTACCOUNT=CMS_COND_21X_GLOBALTAG
GTACCOUNT=$3
#Example: RUNINFOACCOUNT=CMS_COND_31X_RUN_INFO
RUNINFOACCOUNT=$4
#Example: FRONTIER=FrontierProd
FRONTIER=$5
DBTAGCOLLECTION=DBTagsIn_${DB}_${ACCOUNT}.txt
GLOBALTAGCOLLECTION=GlobalTagsForDBAccount.txt
DBTAGDIR=DBTagCollection
GLOBALTAGDIR=GlobalTags
STORAGEPATH=/afs/cern.ch/cms/tracker/sistrcalib/WWW/CondDBMonitoring
WORKDIR=$PWD

#Function to publish png pictures on the web. Will be used at the end of the script:
CreateIndex ()
{
    cp /afs/cern.ch/cms/tracker/sistrcalib/WWW/index_new.html .

    COUNTER=0
    LASTUPDATE=`date`

    for Plot in *.png; do
	if [[ $COUNTER%2 -eq 0 ]]; then
	    cat >> index_new.html  << EOF
<TR> <TD align=center> <a href="$Plot"><img src="$Plot"hspace=5 vspace=5 border=0 style="width: 90%" ALT="$Plot"></a> 
  <br> $Plot </TD>
EOF
	else
	    cat >> index_new.html  << EOF
  <TD align=center> <a href="$Plot"><img src="$Plot"hspace=5 vspace=5 border=0 style="width: 90%" ALT="$Plot"></a> 
  <br> $Plot </TD> </TR> 
EOF
	fi

	let COUNTER++
    done

    cat ${CMSSW_BASE}/src/DQM/SiStripMonitorSummary/data/template_index_foot.html | sed -e "s@insertDate@$LASTUPDATE@g" >> index_new.html

    mv -f index_new.html index.html
}

# Creation of all needed directories if not existing yet
if [ ! -d "$STORAGEPATH/$DB" ]; then 
    afstokenchecker.sh "Creating directory $STORAGEPATH/$DB"
    mkdir $STORAGEPATH/$DB;
fi

if [ ! -d "$STORAGEPATH/$DB/$ACCOUNT" ]; then 
    afstokenchecker.sh "Creating directory $STORAGEPATH/$DB/$ACCOUNT"
    mkdir $STORAGEPATH/$DB/$ACCOUNT; 
fi

if [ ! -d "$STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR" ]; then 
    afstokenchecker.sh "Creating directory $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR"
    mkdir $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR; 
fi

# Array of already analyzed tags

declare -a checkedtags;

# Access of all Global Tags contained in the given DB account
cmscond_tagtree_list -c frontier://cmsfrontier.cern.ch:8000/$FRONTIER/$GTACCOUNT -P /afs/cern.ch/cms/DB/conddb | grep tree | awk '{print $2}' > $GLOBALTAGCOLLECTION

# Definition of some Parameters
MONITOR_QUALITY=True
TAGSUBDIR=RunInfo
LOGDESTINATION=Reader
QUALITYLOGDEST=QualityInfo
CREATETRENDS=True
MONITORCUMULATIVE=False

# Loop on all Global Tags
for globaltag in `cat $GLOBALTAGCOLLECTION`; do

    afstokenchecker.sh "Processing Global Tag $globaltag";

    NEWTAG=False
    NEWIOV=False
    CFGISSAVED=False

    RUNINFOTAGANDOBJECT=`cmscond_tagtree_list -c frontier://cmsfrontier.cern.ch:8000/$FRONTIER/$GTACCOUNT -P /afs/cern.ch/cms/DB/conddb -T $globaltag | grep runinfo | grep $RUNINFOACCOUNT | awk '{printf "%s %s",$3,$5}'`

    if [ `echo $RUNINFOTAGANDOBJECT | wc -w` -eq 0 ]; then
	continue
    fi

    CABLINGTAGANDOBJECT=`cmscond_tagtree_list -c frontier://cmsfrontier.cern.ch:8000/$FRONTIER/$GTACCOUNT -P /afs/cern.ch/cms/DB/conddb -T $globaltag | grep SiStripFedCabling | grep $ACCOUNT | awk '{printf "%s %s",$3,$5}'`

    if [ `echo $CABLINGTAGANDOBJECT | wc -w` -eq 0 ]; then
	continue
    fi

    RUNINFOTAG=`echo $RUNINFOTAGANDOBJECT | awk '{print $1}' | sed -e "s@tag:@@g"`
    RUNINFOOBJECT=`echo $RUNINFOTAGANDOBJECT | awk '{print $2}' | sed -e "s@object:@@g"`

    CABLINGTAG=`echo $CABLINGTAGANDOBJECT | awk '{print $1}' | sed -e "s@tag:@@g"`
    CABLINGOBJECT=`echo $CABLINGTAGANDOBJECT | awk '{print $2}' | sed -e "s@object:@@g"`

    tag=${RUNINFOTAG}_${CABLINGTAG}
    # check if $tag contains blank and if so, issue a warning
    if [ `expr index "$tag" " "` -ne 0 ]; then
	afstokenchecker.sh "WARNING!! $tag has blank spaces"
    fi

    # Creation of DB-Tag directory if not existing yet
    if [ ! -d "$STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR" ]; then 
	afstokenchecker.sh "Creating directory $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR"
	mkdir $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR; 
    fi

    if [ ! -d "$STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag" ]; then 
	afstokenchecker.sh "Creating directory $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag"
	mkdir $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag;

	NEWTAG=True
    fi

    if [ ! -d "$STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/RelatedGlobalTags" ]; then 
	afstokenchecker.sh "Creating directory $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/RelatedGlobalTags"
	mkdir $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/RelatedGlobalTags;
    fi

    # Creation of Global Tag directory if not existing yet
    if [ ! -d "$STORAGEPATH/$DB/$GLOBALTAGDIR" ]; then 
	afstokenchecker.sh "Creating directory $STORAGEPATH/$DB/$GLOBALTAGDIR"
	mkdir $STORAGEPATH/$DB/$GLOBALTAGDIR;
    fi

    if [ ! -d "$STORAGEPATH/$DB/$GLOBALTAGDIR/$globaltag" ]; then 
	afstokenchecker.sh "Creating directory $STORAGEPATH/$DB/$GLOBALTAGDIR/$globaltag"
	mkdir $STORAGEPATH/$DB/$GLOBALTAGDIR/$globaltag;
    fi

    if [ ! -d "$STORAGEPATH/$DB/$GLOBALTAGDIR/$globaltag/RunInfo" ]; then 
	afstokenchecker.sh "Creating directory $STORAGEPATH/$DB/$GLOBALTAGDIR/$globaltag/RunInfo"
	mkdir $STORAGEPATH/$DB/$GLOBALTAGDIR/$globaltag/RunInfo;
    fi

    # Creation of links between the DB-Tag and the respective Global Tags
    if [ ! -f $STORAGEPATH/$DB/$GLOBALTAGDIR/$globaltag/RunInfo/$tag ] || [ ! -f $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/RelatedGlobalTags/$globaltag ]; then
	cd $STORAGEPATH/$DB/$GLOBALTAGDIR/$globaltag/RunInfo;
	rm -f $tag;
	cat >> $tag << EOF
<html>
<body>
<a href="https://test-stripdbmonitor.web.cern.ch/test-stripdbmonitor/CondDBMonitoring/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag">https://test-stripdbmonitor.web.cern.ch/test-stripdbmonitor/CondDBMonitoring/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag</a>
</body>
</html>
EOF

	cd $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/RelatedGlobalTags;
	rm -f $globaltag;
	cat >> $globaltag << EOF
<html>
<body>
<a href="https://test-stripdbmonitor.web.cern.ch/test-stripdbmonitor/CondDBMonitoring/$DB/$GLOBALTAGDIR/$globaltag">https://test-stripdbmonitor.web.cern.ch/test-stripdbmonitor/CondDBMonitoring/$DB/$GLOBALTAGDIR/$globaltag</a>
</body>
</html>
EOF

    cd $WORKDIR;

# check if the tag has been analyzed already

    ALREADYCHECKED=0;

    for checkedtag in ${checkedtags[*]}; do
	if [ $checkedtag == $tag ]; then
	    ALREADYCHECKED=1
	fi
    done

    if [ $ALREADYCHECKED -eq 1 ]; then
	date "+[%c] Tags $tag already checked: skip"
	continue
    fi

    checkedtags[${#checkedtags[*]}]=$tag;

    # Get the list of IoVs for the given DB-Tag
    iov_list_tag.py -c frontier://cmsfrontier.cern.ch:8000/$FRONTIER/$RUNINFOACCOUNT -P /afs/cern.ch/cms/DB/conddb -t $RUNINFOTAG  > list_Iov.txt # Access via Frontier

    # Access DB for the given DB-Tag and dump histograms in .png if not existing yet
    afstokenchecker.sh "Now the values are retrieved from the DB..."
    
    if [ ! -d "$STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/rootfiles" ]; then
	mkdir $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/rootfiles;
	mkdir $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/cfg;
	mkdir $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots;
	mkdir $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/TIB;
	mkdir $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/TOB;
	mkdir $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/TID;
	mkdir $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/TEC;

	for i in {1..4}; do
	    mkdir $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/TIB/Layer$i;

	    if [ "$MONITORCUMULATIVE" = "True" ] || [ "$CREATETRENDS" = "True" ]; then
		mkdir $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/TIB/Layer$i/Profile;
	    fi
	    if [ "$MONITORCUMULATIVE" = "True" ]; then
		mkdir $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/TIB/Layer$i/Cumulative;
	    fi
	    if [ "$CREATETRENDS" = "True" ]; then
		mkdir $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/TIB/Layer$i/Trends;
	    fi
	done

	for i in {1..6}; do
	    mkdir $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/TOB/Layer$i;

	    if [ "$MONITORCUMULATIVE" = "True" ] || [ "$CREATETRENDS" = "True" ]; then
		mkdir $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/TOB/Layer$i/Profile;
	    fi
	    if [ "$MONITORCUMULATIVE" = "True" ]; then
		mkdir $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/TOB/Layer$i/Cumulative;
	    fi
	    if [ "$CREATETRENDS" = "True" ]; then
		mkdir $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/TOB/Layer$i/Trends;
	    fi
	done

	for i in {1..2}; do
	    mkdir $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/TID/Side$i;
	    for j in {1..3}; do
		mkdir $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/TID/Side$i/Disk$j;

		if [ "$MONITORCUMULATIVE" = "True" ] || [ "$CREATETRENDS" = "True" ]; then
		    mkdir $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/TID/Side$i/Disk$j/Profile;
		fi
		if [ "$MONITORCUMULATIVE" = "True" ]; then
		    mkdir $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/TID/Side$i/Disk$j/Cumulative;
		fi
		if [ "$CREATETRENDS" = "True" ]; then
		    mkdir $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/TID/Side$i/Disk$j/Trends;
		fi
	    done
	done

	for i in {1..2}; do
	    mkdir $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/TEC/Side$i;
	    for j in {1..9}; do
		mkdir $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/TEC/Side$i/Disk$j;

		if [ "$MONITORCUMULATIVE" = "True" ] || [ "$CREATETRENDS" = "True" ]; then
		    mkdir $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/TEC/Side$i/Disk$j/Profile;
		fi
		if [ "$MONITORCUMULATIVE" = "True" ]; then
		    mkdir $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/TEC/Side$i/Disk$j/Cumulative;
		fi
		if [ "$CREATETRENDS" = "True" ]; then
		    mkdir $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/TEC/Side$i/Disk$j/Trends;
		fi
	    done
	done

	if [ "$MONITOR_QUALITY" = "True" ] || [ "$MONITOR_CABLING" = "True" ]; then
	    mkdir $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/Trends
	    mkdir $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/TIB/Trends;
	    mkdir $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/TOB/Trends;
	    mkdir $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/TID/Side1/Trends;
	    mkdir $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/TID/Side2/Trends;
	    mkdir $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/TEC/Side1/Trends;
	    mkdir $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/TEC/Side2/Trends;
	    mkdir $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/Summary
	fi


	mkdir $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/TrackerMap;

	if [ "$MONITOR_QUALITY" = "True" ]; then
	    mkdir $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/Summary/BadAPVs
	    mkdir $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/Summary/BadFibers
	    mkdir $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/Summary/BadModules
	    mkdir $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/Summary/BadStrips
	    mkdir $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/QualityLog
	fi

	if [ "$MONITOR_CABLING" = "True" ]; then
	    mkdir $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/CablingLog
	fi

    fi

    if [ `echo *.png | wc -w` -gt 0 ]; then
	rm *.png;
    fi

    # Produce only the cfg. The whole Monitoring of all IOVs is too much!!!
    if [ "$NEWTAG" = "True" ]; then # Skip Tags already processed. Take only new ones.
	afstokenchecker.sh "New Tag $tag found. Being processed..."
	CMSRUNCOMMAND="cmsRun ${CMSSW_BASE}/src/DQM/SiStripMonitorSummary/test/RunInfo_conddbmonitoring_cfg.py print logDestination=$LOGDESTINATION qualityLogDestination=$QUALITYLOGDEST runInfoTag=$RUNINFOTAG cablingTag=$CABLINGTAG cablingConnectionString=frontier://$FRONTIER/$ACCOUNT runinfoConnectionString=frontier://$FRONTIER/$RUNINFOACCOUNT MonitorCumulative=$MONITORCUMULATIVE"  
#outputRootFile=$ROOTFILE 
#runNumber=$IOV_number

	cp ${CMSSW_BASE}/src/DQM/SiStripMonitorSummary/test/RunInfo_conddbmonitoring_cfg.py RunInfo_cfg.py 
	cat >> RunInfo_cfg.py <<EOF
#
# $CMSRUNCOMMAND
#
EOF
	mv RunInfo_cfg.py $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/cfg/${tag}_cfg.py
	continue
    else
	continue
    fi

#    # Process each IOV of the given DB-Tag seperately
#    for IOV_number in `cat list_Iov.txt`; do
#
#	if [ "$IOV_number" = "Total" ] || [ $IOV_number -gt 100000000 ]; then
#	    continue
#	fi
#
#	ROOTFILE="${tag}_Run_${IOV_number}.root"
#
#	if [ -f $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/rootfiles/$ROOTFILE ]; then # Skip IOVs already processed. Take only new ones.
#	    continue
#	fi
#
#	afstokenchecker.sh "New IOV $IOV_number found. Being processed..."
#
#	NEWIOV=True
#
#	cat template_DBReaderRunInfo_cfg.py | sed -e "s@insertRun@$IOV_number@g" -e "s@insertLog@$LOGDESTINATION@g" -e "s@insertDB@$DB@g" -e "s@insertFrontier@$FRONTIER@g" -e "s@insertInfoAccount@$RUNINFOACCOUNT@g" -e "s@insertCablingAccount@$ACCOUNT@g" -e "s@insertInfoTag@$RUNINFOTAG@g" -e "s@insertCablingTag@$CABLINGTAG@g" -e "s@insertOutFile@$ROOTFILE@g" -e "s@insertMonitorCumulative@$MONITORCUMULATIVE@g"> DBReader_cfg.py
#
#	afstokenchecker.sh "Executing cmsRun. Stay tuned ..."
#
#	cmsRun DBReader_cfg.py
#
#	afstokenchecker.sh "cmsRun finished. Now moving the files to the corresponding directories ..."
#
#	if [ "$NEWTAG" = "True" ] && [ "$CFGISSAVED" = "False" ]; then
#	    cp DBReader_cfg.py $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/cfg/${tag}_cfg.py
#	    CFGISSAVED=True
#	fi
#
#	mv $ROOTFILE $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/rootfiles;
#
#	if [ "$MONITOR_QUALITY" = "True" ]; then
#	    cat $LOGDESTINATION.log | awk 'BEGIN{doprint=0}{if(match($0,"New IOV")!=0) doprint=1;if(match($0,"%MSG")!=0) {doprint=0;} if(doprint==1) print $0}' > QualityInfo_Run${IOV_number}.txt
#	    mv QualityInfo_Run${IOV_number}.txt $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/QualityLog/
#
#	    rm $LOGDESTINATION.log
#	fi
#
#	if [ "$MONITOR_CABLING" = "True" ]; then
#	    if [ "$ACCOUNT" != "CMS_COND_21X_STRIP" ]; then
#		cat $LOGDESTINATION.log | awk 'BEGIN{doprint=0}{if(match($0,"beginRun")!=0) doprint=1;if(match($0,"%MSG")!=0) {doprint=0;} if(doprint==1) print $0}' > CablingInfo_Run${IOV_number}.txt
#		mv CablingInfo_Run${IOV_number}.txt $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/CablingLog/
#	    fi
#
#	    cat $LOGDESTINATION.log | awk 'BEGIN{doprint=0}{if(match($0,"New IOV")!=0) doprint=1;if(match($0,"%MSG")!=0) {doprint=0;} if(doprint==1) print $0}' > QualityInfoFromCabling_Run${IOV_number}.txt
#	    mv QualityInfoFromCabling_Run${IOV_number}.txt $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/CablingLog/
#
#	    rm $LOGDESTINATION.log
#	fi
#
#	for Plot in `ls *.png | grep TIB`; do
#	    PNGNAME=`echo ${Plot#*_*_*_*_*_} | gawk -F . '{print $1}'`
#	    LAYER=`echo ${PNGNAME#*_*_} | gawk -F _ '{print $1}'`
#	    if [ "$MONITORCUMULATIVE" = "True" ] || [ "$CREATETRENDS" = "True" ]; then
#		if [ `echo $Plot | grep Cumulative` ]; then
#		    mv $Plot $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/TIB/Layer$LAYER/Cumulative/${PNGNAME}__Run${IOV_number}.png;
#		else
#		    mv $Plot $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/TIB/Layer$LAYER/Profile/${PNGNAME}__Run${IOV_number}.png;
#		fi
#	    else
#		mv $Plot $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/TIB/Layer$LAYER/${PNGNAME}__Run${IOV_number}.png;
#	    fi
#	done;
#
#	for Plot in `ls *.png | grep TOB`; do
#	    PNGNAME=`echo ${Plot#*_*_*_*_*_} | gawk -F . '{print $1}'`
#	    LAYER=`echo ${PNGNAME#*_*_} | gawk -F _ '{print $1}'`
#	    if [ "$MONITORCUMULATIVE" = "True" ] || [ "$CREATETRENDS" = "True" ]; then
#		if [ `echo $Plot | grep Cumulative` ]; then
#		    mv $Plot $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/TOB/Layer$LAYER/Cumulative/${PNGNAME}__Run${IOV_number}.png;
#		else
#		    mv $Plot $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/TOB/Layer$LAYER/Profile/${PNGNAME}__Run${IOV_number}.png;
#		fi
#	    else
#		mv $Plot $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/TOB/Layer$LAYER/${PNGNAME}__Run${IOV_number}.png;
#	    fi
#	done;
#
#	for Plot in `ls *.png | grep TID`; do
#	    PNGNAME=`echo ${Plot#*_*_*_*_*_} | gawk -F . '{print $1}'`
#	    SIDE=`echo ${PNGNAME#*_*_} | gawk -F _ '{print $1}'`
#	    DISK=`echo ${PNGNAME#*_*_*_*_*_*_} | gawk -F _ '{print $1}'`
#	    if [ "$MONITORCUMULATIVE" = "True" ] || [ "$CREATETRENDS" = "True" ]; then
#		if [ `echo $Plot | grep Cumulative` ]; then
#		    mv $Plot $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/TID/Side$SIDE/Disk$DISK/Cumulative/${PNGNAME}__Run${IOV_number}.png;
#		else
#		    mv $Plot $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/TID/Side$SIDE/Disk$DISK/Profile/${PNGNAME}__Run${IOV_number}.png;
#		fi
#	    else
#		mv $Plot $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/TID/Side$SIDE/Disk$DISK/${PNGNAME}__Run${IOV_number}.png;
#	    fi
#	done;
#
#	for Plot in `ls *.png | grep TEC`; do
#	    PNGNAME=`echo ${Plot#*_*_*_*_*_} | gawk -F . '{print $1}'`
#	    SIDE=`echo ${PNGNAME#*_*_} | gawk -F _ '{print $1}'`
#	    DISK=`echo ${PNGNAME#*_*_*_*_*_*_} | gawk -F _ '{print $1}'`
#	    if [ "$MONITORCUMULATIVE" = "True" ] || [ "$CREATETRENDS" = "True" ]; then
#		if [ `echo $Plot | grep Cumulative` ]; then
#		    mv $Plot $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/TEC/Side$SIDE/Disk$DISK/Cumulative/${PNGNAME}__Run${IOV_number}.png;
#		else
#		    mv $Plot $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/TEC/Side$SIDE/Disk$DISK/Profile/${PNGNAME}__Run${IOV_number}.png;
#		fi
#	    else
#		mv $Plot $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/TEC/Side$SIDE/Disk$DISK/${PNGNAME}__Run${IOV_number}.png;
#	    fi
#	done;
#
#	for Plot in `ls *.png | grep TkMap`; do
#	    #PNGNAME=`echo $Plot | gawk -F . '{print $1}'`
#	    mv $Plot $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/TrackerMap/$Plot;
#	done;
#
#	for Plot in `ls *.png | grep Bad`; do
#	    PNGNAME=`echo ${Plot#*_} | gawk -F . '{print $1}'`
#	    if      [ `echo $PNGNAME | grep Apvs` ]; then
#		mv $Plot $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/Summary/BadAPVs/${PNGNAME}__Run${IOV_number}.png;
#	    else if [ `echo $PNGNAME | grep Fibers` ]; then
#		mv $Plot $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/Summary/BadFibers/${PNGNAME}__Run${IOV_number}.png;
#	    else if [ `echo $PNGNAME | grep Modules` ]; then
#		mv $Plot $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/Summary/BadModules/${PNGNAME}__Run${IOV_number}.png;
#	    else if [ `echo $PNGNAME | grep Strips` ]; then
#		mv $Plot $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/Summary/BadStrips/${PNGNAME}__Run${IOV_number}.png;
#	    fi
#	    fi
#	    fi
#	    fi
#	done;
#
#	for Plot in `ls *.png | grep Cabling`; do
#	    PNGNAME=`echo ${Plot#*_} | gawk -F . '{print $1}'`
#	    mv $Plot $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/Summary/${PNGNAME}__Run${IOV_number}.png;
#	done;
#
#	#cd $WORKDIR;
#
#    done;

#    # Run the Trends and Publish all histograms on a web page
#    if [ "$NEWTAG" = "True" ] || [ "$NEWIOV" = "True" ]; then
#
#	if [ "$CREATETRENDS" = "True" ]; then
#	    afstokenchecker.sh "Creating the Trend Plots ..."
#
#	    ./getOfflineDQMData.sh $DB $ACCOUNT $TAGSUBDIR $tag
#	    getOfflineDQMData.sh $DB $ACCOUNT $TAGSUBDIR $tag
#
#	    for i in {1..4}; do
#		for Plot in `ls *.png | grep TIBLayer$i`; do
#		    mv $Plot $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/TIB/Layer$i/Trends;
#		done
#	    done
#
#	    for i in {1..6}; do
#		for Plot in `ls *.png | grep TOBLayer$i`; do
#		    mv $Plot $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/TOB/Layer$i/Trends;
#		done
#	    done
#
#	    for i in {1..3}; do
#		for Plot in `ls *.png | grep TID-Disk$i`; do
#		    mv $Plot $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/TID/Side1/Disk$i/Trends;
#		done
#		for Plot in `ls *.png | grep TID+Disk$i`; do
#		    mv $Plot $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/TID/Side2/Disk$i/Trends;
#		done
#	    done
#
#	    for i in {1..9}; do
#		for Plot in `ls *.png | grep TEC-Disk$i`; do
#		    mv $Plot $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/TEC/Side1/Disk$i/Trends;
#		done
#		for Plot in `ls *.png | grep TEC+Disk$i`; do
#		    mv $Plot $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/TEC/Side2/Disk$i/Trends;
#		done
#	    done
#	    
#	    for Plot in `ls *.png | grep TIB`; do
#		mv $Plot $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/TIB/Trends;
#	    done
#
#	    for Plot in `ls *.png | grep TOB`; do
#		mv $Plot $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/TOB/Trends;
#	    done
#
#	    for Plot in `ls *.png | grep TID-`; do
#		mv $Plot $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/TID/Side1/Trends;
#	    done
#
#	    for Plot in `ls *.png | grep TID+`; do
#		mv $Plot $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/TID/Side2/Trends;
#	    done
#
#	    for Plot in `ls *.png | grep TEC-`; do
#		mv $Plot $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/TEC/Side1/Trends;
#	    done
#
#	    for Plot in `ls *.png | grep TEC+`; do
#		mv $Plot $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/TEC/Side2/Trends;
#	    done
#
#	    for Plot in `ls *.png | grep Tracker`; do
#		mv $Plot $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/Trends;
#	    done
#	fi
#
#	mv TrackerSummary.root $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/rootfiles;
#	rm -f TrackerPlots.root;
#	rm -f makePlots_cc.d makePlots_cc.so;
#	rm -f makeTKTrend_cc.d makeTKTrend_cc.so;
#
#	afstokenchecker.sh "Publishing the new tag $tag (or the new IOV) on the web ..."
#
#	for i in {1..4}; do
#	    cd /afs/cern.ch/cms/tracker/sistrcalib/WWW;
#	    cat ${CMSSW_BASE}/src/DQM/SiStripMonitorSummary/data/template_index_header.html  | sed -e "s@insertPageName@$tag --- TIB Layer $i --- Summary Report@g" > index_new.html
#	    if [ "$MONITORCUMULATIVE" = "True" ] || [ "$CREATETRENDS" = "True" ]; then
#		cd $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/TIB/Layer$i/Profile;
#		CreateIndex
#
#		if [ "$MONITORCUMULATIVE" = "True" ]; then
#		    cd $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/TIB/Layer$i/Cumulative;
#		    CreateIndex
#		fi
#
#		if [ "$CREATETRENDS" = "True" ]; then
#		    cd $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/TIB/Layer$i/Trends;
#		    CreateIndex
#		fi
#	    else
#		cd $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/TIB/Layer$i;
#		CreateIndex
#	    fi
#	done
#
#	for i in {1..6}; do
#	    cd /afs/cern.ch/cms/tracker/sistrcalib/WWW;
#	    cat ${CMSSW_BASE}/src/DQM/SiStripMonitorSummary/data/template_index_header.html  | sed -e "s@insertPageName@$tag --- TOB Layer $i --- Summary Report@g" > index_new.html
#	    if [ "$MONITORCUMULATIVE" = "True" ] || [ "$CREATETRENDS" = "True" ]; then
#		cd $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/TOB/Layer$i/Profile;
#		CreateIndex
#
#		if [ "$MONITORCUMULATIVE" = "True" ]; then
#		    cd $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/TOB/Layer$i/Cumulative;
#		    CreateIndex
#		fi
#
#		if [ "$CREATETRENDS" = "True" ]; then
#		    cd $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/TOB/Layer$i/Trends;
#		    CreateIndex
#		fi
#	    else
#		cd $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/TOB/Layer$i;
#		CreateIndex
#	    fi
#	done
#
#	for i in {1..2}; do
#	    for j in {1..3}; do
#		cd /afs/cern.ch/cms/tracker/sistrcalib/WWW;
#		cat ${CMSSW_BASE}/src/DQM/SiStripMonitorSummary/data/template_index_header.html  | sed -e "s@insertPageName@$tag --- TID Side $i Disk $j --- Summary Report@g" > index_new.html
#		if [ "$MONITORCUMULATIVE" = "True" ] || [ "$CREATETRENDS" = "True" ]; then
#		    cd $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/TID/Side$i/Disk$j/Profile;
#		    CreateIndex
#
#		    if [ "$MONITORCUMULATIVE" = "True" ]; then
#			cd $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/TID/Side$i/Disk$j/Cumulative;
#			CreateIndex
#		    fi
#
#		    if [ "$CREATETRENDS" = "True" ]; then
#			cd $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/TID/Side$i/Disk$j/Trends;
#			CreateIndex
#		    fi
#		else
#		    cd $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/TID/Side$i/Disk$j;
#		    CreateIndex
#		fi
#	    done
#	done
#
#	for i in {1..2}; do
#	    for j in {1..9}; do
#		cd /afs/cern.ch/cms/tracker/sistrcalib/WWW;
#		cat ${CMSSW_BASE}/src/DQM/SiStripMonitorSummary/data/template_index_header.html  | sed -e "s@insertPageName@$tag --- TEC Side $i Disk $j --- Summary Report@g" > index_new.html
#		if [ "$MONITORCUMULATIVE" = "True" ] || [ "$CREATETRENDS" = "True" ]; then
#		    cd $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/TEC/Side$i/Disk$j/Profile;
#		    CreateIndex
#
#		    if [ "$MONITORCUMULATIVE" = "True" ]; then
#			cd $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/TEC/Side$i/Disk$j/Cumulative;
#			CreateIndex
#		    fi
#
#		    if [ "$CREATETRENDS" = "True" ]; then
#			cd $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/TEC/Side$i/Disk$j/Trends;
#			CreateIndex
#		    fi
#		else
#		    cd $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/TEC/Side$i/Disk$j;
#		    CreateIndex
#		fi
#	    done
#	done
#
#	if [ "$CREATETRENDS" = "True" ]; then
#	    cd /afs/cern.ch/cms/tracker/sistrcalib/WWW;
#	    cat ${CMSSW_BASE}/src/DQM/SiStripMonitorSummary/data/template_index_header.html  | sed -e "s@insertPageName@$tag --- Full Strip Tracker --- Trend Plots@g" > index_new.html
#	    cd $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/Trends;
#	    CreateIndex
#
#	    cd /afs/cern.ch/cms/tracker/sistrcalib/WWW;
#	    cat ${CMSSW_BASE}/src/DQM/SiStripMonitorSummary/data/template_index_header.html  | sed -e "s@insertPageName@$tag --- TIB --- Trend Plots@g" > index_new.html
#	    cd $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/TIB/Trends;
#	    CreateIndex
#
#	    cd /afs/cern.ch/cms/tracker/sistrcalib/WWW;
#	    cat ${CMSSW_BASE}/src/DQM/SiStripMonitorSummary/data/template_index_header.html  | sed -e "s@insertPageName@$tag --- TOB --- Trend Plots@g" > index_new.html
#	    cd $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/TOB/Trends;
#	    CreateIndex
#
#	    for i in {1..2}; do
#		cd /afs/cern.ch/cms/tracker/sistrcalib/WWW;
#		cat ${CMSSW_BASE}/src/DQM/SiStripMonitorSummary/data/template_index_header.html  | sed -e "s@insertPageName@$tag --- TID Side $i --- Trend Plots@g" > index_new.html
#		cd $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/TID/Side$i/Trends;
#		CreateIndex
#
#		cd /afs/cern.ch/cms/tracker/sistrcalib/WWW;
#		cat ${CMSSW_BASE}/src/DQM/SiStripMonitorSummary/data/template_index_header.html  | sed -e "s@insertPageName@$tag --- TEC Side $i --- Trend Plots@g" > index_new.html
#		cd $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/TEC/Side$i/Trends;
#		CreateIndex
#	    done
#	fi
#
#	if [ "$MONITOR_QUALITY" = "True" ]; then
#	    cd /afs/cern.ch/cms/tracker/sistrcalib/WWW;
#	    cat ${CMSSW_BASE}/src/DQM/SiStripMonitorSummary/data/template_index_header.html  | sed -e "s@insertPageName@$tag --- Bad APVs --- Summary Report@g" > index_new.html
#	    cd $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/Summary/BadAPVs;
#	    CreateIndex
#
#	    cd /afs/cern.ch/cms/tracker/sistrcalib/WWW;
#	    cat ${CMSSW_BASE}/src/DQM/SiStripMonitorSummary/data/template_index_header.html  | sed -e "s@insertPageName@$tag --- Bad Fibers --- Summary Report@g" > index_new.html
#	    cd $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/Summary/BadFibers;
#	    CreateIndex
#	    
#	    cd /afs/cern.ch/cms/tracker/sistrcalib/WWW;
#	    cat ${CMSSW_BASE}/src/DQM/SiStripMonitorSummary/data/template_index_header.html  | sed -e "s@insertPageName@$tag --- Bad Modules --- Summary Report@g" > index_new.html
#	    cd $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/Summary/BadModules;
#	    CreateIndex
#
#	    cd /afs/cern.ch/cms/tracker/sistrcalib/WWW;
#	    cat ${CMSSW_BASE}/src/DQM/SiStripMonitorSummary/data/template_index_header.html  | sed -e "s@insertPageName@$tag --- Bad Strips --- Summary Report@g" > index_new.html
#	    cd $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/Summary/BadStrips;
#	    CreateIndex
#		
#	fi
#    
#	if [ "$MONITOR_CABLING" = "True" ]; then
#	    cd /afs/cern.ch/cms/tracker/sistrcalib/WWW;
#	    cat ${CMSSW_BASE}/src/DQM/SiStripMonitorSummary/data/template_index_header.html  | sed -e "s@insertPageName@$tag --- Summary Report@g" > index_new.html
#	    cd $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/Summary/;
#	    CreateIndex
#		
#	fi
#
#	if [ -d "$STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/TrackerMap" ]; then
#	    cd /afs/cern.ch/cms/tracker/sistrcalib/WWW;
#	    cat ${CMSSW_BASE}/src/DQM/SiStripMonitorSummary/data/template_index_header.html  | sed -e "s@insertPageName@$tag --- Tracker Maps for all IOVs ---@g" > index_new.html
#	    cd $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/TrackerMap;
#	    CreateIndex
#
#	fi
#
#    fi
#
#    cd $WORKDIR;

done;
