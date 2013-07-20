#!/bin/bash

date

if [ $# -ne 4 ]; then
    echo "You have to provide a <DB>, an <Account>, a <GTAccount> and the <FrontierPath> !!!"
    exit
fi

#Example: DB=cms_orcoff_prod
DB=$1
#Example: ACCOUNT=CMS_COND_21X_STRIP
ACCOUNT=$2
#Example: GTACCOUNT=CMS_COND_21X_GLOBALTAG
GTACCOUNT=$3
#Example: FRONTIER=FrontierProd
FRONTIER=$4
DBTAGCOLLECTION=DBTagsIn_${DB}_${ACCOUNT}.txt
GLOBALTAGCOLLECTION=GlobalTagsForDBTag.txt
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

    for Plot in `ls *.png`; do
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

    cat /afs/cern.ch/cms/tracker/sistrcalib/WWW/template_index_foot.html | sed -e "s@insertDate@$LASTUPDATE@g" >> index_new.html

    mv -f index_new.html index.html
}

# Creation of all needed directories if not existing yet
if [ ! -d "$STORAGEPATH/$DB" ]; then 
    echo "Creating directory $STORAGEPATH/$DB"
    mkdir $STORAGEPATH/$DB;
fi

if [ ! -d "$STORAGEPATH/$DB/$ACCOUNT" ]; then 
    echo "Creating directory $STORAGEPATH/$DB/$ACCOUNT"
    mkdir $STORAGEPATH/$DB/$ACCOUNT; 
fi

if [ ! -d "$STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR" ]; then 
    echo "Creating directory $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR"
    mkdir $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR; 
fi

# Creation of Global Tag directory from scratch to have always up-to-date information (and no old links that are obsolete)
if [ -d "$STORAGEPATH/$DB/$ACCOUNT/$GLOBALTAGDIR" ]; then 
    rm -rf $STORAGEPATH/$DB/$ACCOUNT/$GLOBALTAGDIR; 
fi

echo "Creating directory $STORAGEPATH/$DB/$ACCOUNT/$GLOBALTAGDIR"
mkdir $STORAGEPATH/$DB/$ACCOUNT/$GLOBALTAGDIR; 

# Access of all SiStrip Tags uploaded to the given DB account
#cmscond_list_iov -c oracle://$DB/$ACCOUNT -P /afs/cern.ch/cms/DB/conddb | grep SiStrip > $DBTAGCOLLECTION # Access via oracle
cmscond_list_iov -c frontier://$FRONTIER/$ACCOUNT -P /afs/cern.ch/cms/DB/conddb | grep SiStrip > $DBTAGCOLLECTION # Access via Frontier

# Loop on all DB Tags
for tag in `cat $DBTAGCOLLECTION`; do

    echo "Processing DB-Tag $tag";

    NEWTAG=False
    NEWIOV=False
    CFGISSAVED=False

    # Discover which kind of tag is processed
    MONITOR_NOISE=False
    MONITOR_PEDESTAL=False
    MONITOR_GAIN=False
    MONITOR_QUALITY=False
    MONITOR_CABLING=False
    MONITOR_LA=False
    MONITOR_THRESHOLD=False

    LOGDESTINATION=cout

    MONITORCUMULATIVE=False
    CREATETRENDS=False

    if      [ `echo $tag | grep "Noise" | wc -w` -gt 0 ]; then
	MONITOR_NOISE=True
	USEACTIVEDETID=True
	RECORD=SiStripNoisesRcd
	TAGSUBDIR=SiStripNoise
	MONITORCUMULATIVE=True
    else if [ `echo $tag | grep "Pedestal" | wc -w` -gt 0 ]; then
	MONITOR_PEDESTAL=True
	USEACTIVEDETID=True
	RECORD=SiStripPedestalsRcd
	TAGSUBDIR=SiStripPedestal
    else if [ `echo $tag | grep "Gain" | wc -w` -gt 0 ]; then
	MONITOR_GAIN=True
	USEACTIVEDETID=True
	RECORD=SiStripApvGainRcd
	TAGSUBDIR=SiStripApvGain
    else if [ `echo $tag | grep "Bad" | wc -w` -gt 0 ]; then
	MONITOR_QUALITY=True
	USEACTIVEDETID=True
	RECORD=SiStripBadChannelRcd
	TAGSUBDIR=SiStripBadChannel
	LOGDESTINATION=Reader
	CREATETRENDS=True
    else if [ `echo $tag | grep "Cabling" | wc -w` -gt 0 ]; then
	MONITOR_CABLING=True
	USEACTIVEDETID=True
	RECORD=SiStripFedCablingRcd
	RECORDFORQUALITY=SiStripDetCablingRcd
	TAGSUBDIR=SiStripFedCabling
	LOGDESTINATION=Reader
	CREATETRENDS=True
    else if [ `echo $tag | grep "Lorentz" | wc -w` -gt 0 ]; then
	MONITOR_LA=True
	USEACTIVEDETID=False
	RECORD=SiStripLorentzAngleRcd
	TAGSUBDIR=SiStripLorentzAngle
    else if [ `echo $tag | grep "Threshold" | wc -w` -gt 0 ]; then
	MONITOR_THRESHOLD=True
	USEACTIVEDETID=True
	RECORD=SiStripThresholdRcd
	TAGSUBDIR=SiStripThreshold
    else if [ `echo $tag | grep "VOff" | wc -w` -gt 0 ]; then
	MONITOR_QUALITY=True
	USEACTIVEDETID=True
	RECORD=SiStripDetVOffRcd
	TAGSUBDIR=SiStripVoltage
	LOGDESTINATION=Reader
	CREATETRENDS=True
    else
	USEACTIVEDETID=False
	RECORD=Unknown
	TAGSUBDIR=Unknown

    fi
    fi
    fi
    fi
    fi
    fi
    fi
    fi

    # Creation of DB-Tag directory if not existing yet
    if [ ! -d "$STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR" ]; then 
	echo "Creating directory $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR"
	mkdir $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR; 
    fi

    if [ ! -d "$STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag" ]; then 
	echo "Creating directory $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag"
	mkdir $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag;

	NEWTAG=True
    fi

    if [ -d "$STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/RelatedGlobalTags" ]; then
	rm -rf $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/RelatedGlobalTags; # remove former links to be safe if something has changed there
    fi

    mkdir $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/RelatedGlobalTags; # start from scratch to have always the up-to-date information

    # Access of all Global Tags for the given DB Tag
    if [ -f globaltag_tmp.txt ]; then
	rm globaltag_tmp.txt;
    fi

    if [ -f $GLOBALTAGCOLLECTION ]; then
	rm $GLOBALTAGCOLLECTION;
    fi

    if [ `echo $DB | grep "prep" | wc -w` -gt 0 ]; then
	for globaltag in `cmscond_tagintrees -c sqlite_file:$WORKDIR/CMSSW_3_2_5/src/CondCore/TagCollection/data/GlobalTag.db -P /afs/cern.ch/cms/DB/conddb -t $tag | grep Trees`; do # Access via Frontier

	    if [ "$globaltag" != "#" ] && [ "$globaltag" != "Trees" ]; then
		echo $globaltag >> globaltag_tmp.txt;
		cat globaltag_tmp.txt | sed -e "s@\[@@g" -e "s@\]@@g" -e "s@'@@g" -e "s@,@@g" > $GLOBALTAGCOLLECTION
	    fi

	done;

    else
	for globaltag in `cmscond_tagintrees -c frontier://cmsfrontier.cern.ch:8000/$FRONTIER/$GTACCOUNT -P /afs/cern.ch/cms/DB/conddb -t $tag | grep Trees`; do # Access via Frontier

	    if [ "$globaltag" != "#" ] && [ "$globaltag" != "Trees" ]; then
		echo $globaltag >> globaltag_tmp.txt;
		cat globaltag_tmp.txt | sed -e "s@\[@@g" -e "s@\]@@g" -e "s@'@@g" -e "s@,@@g" > $GLOBALTAGCOLLECTION
	    fi

	done;
    fi

    # Loop on all Global Tags
    if [ -f $GLOBALTAGCOLLECTION ]; then
	for globaltag in `cat $GLOBALTAGCOLLECTION`; do

	    echo "Processing Global Tag $globaltag"

	# Creation of Global Tag directory if not existing yet
	    if [ ! -d "$STORAGEPATH/$DB/$ACCOUNT/$GLOBALTAGDIR/$globaltag" ]; then 
		echo "Creating directory $STORAGEPATH/$DB/$ACCOUNT/$GLOBALTAGDIR/$globaltag"
		mkdir $STORAGEPATH/$DB/$ACCOUNT/$GLOBALTAGDIR/$globaltag;
	    fi

	# Creation of links between the DB-Tag and the respective Global Tags
	    cd $STORAGEPATH/$DB/$ACCOUNT/$GLOBALTAGDIR/$globaltag;
	    if [ -f $tag ]; then
		rm $tag;
	    fi
	    cat >> $tag << EOF
<html>
<body>
<a href="https://test-stripdbmonitor.web.cern.ch/test-stripdbmonitor/CondDBMonitoring/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag">https://test-stripdbmonitor.web.cern.ch/test-stripdbmonitor/CondDBMonitoring/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag</a>
</body>
</html>
EOF
	    #ln -s $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag $tag;
	    cd $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/RelatedGlobalTags;
	    if [ -f $globaltag ]; then
		rm $globaltag;
	    fi
	    cat >> $globaltag << EOF
<html>
<body>
<a href="https://test-stripdbmonitor.web.cern.ch/test-stripdbmonitor/CondDBMonitoring/$DB/$ACCOUNT/$GLOBALTAGDIR/$globaltag">https://test-stripdbmonitor.web.cern.ch/test-stripdbmonitor/CondDBMonitoring/$DB/$ACCOUNT/$GLOBALTAGDIR/$globaltag</a>
</body>
</html>
EOF
	    #ln -s $STORAGEPATH/$DB/$ACCOUNT/$GLOBALTAGDIR/$globaltag $globaltag;
	    cd $WORKDIR;

	done;

    fi

    if [ "$RECORD" = "Unknown" ]; then
	echo "Unknown strip tag. Processing skipped!"
	continue
    fi

    # Get the list of IoVs for the given DB-Tag
    #cmscond_list_iov -c oracle://$DB/$ACCOUNT -P /afs/cern.ch/cms/DB/conddb -t $tag | awk '{if(NR>4) print "Run_In "$1 " Run_End " $2}' > list_Iov.txt  # Access via oracle
    cmscond_list_iov -c frontier://$FRONTIER/$ACCOUNT -P /afs/cern.ch/cms/DB/conddb -t $tag | awk '{if(NR>4) print "Run_In "$1 " Run_End " $2}' > list_Iov.txt # Access via Frontier

    # Access DB for the given DB-Tag and dump values in a root-file and histograms in .png if not existing yet
    echo "Now the values are retrieved from the DB..."
    
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

    if [ `ls *.png | wc -w` -gt 0 ]; then
	rm *.png;
    fi

    # Process each IOV of the given DB-Tag seperately
    for IOV_number in `grep Run_In list_Iov.txt | awk '{print $2}'`; do

	if [ "$IOV_number" = "Total" ] || [ $IOV_number -gt 100000000 ]; then
	    continue
	fi
	
	ROOTFILE="${tag}_Run_${IOV_number}.root"

	if [ -f $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/rootfiles/$ROOTFILE ]; then # Skip IOVs already processed. Take only new ones.
	    continue
	fi

	echo "New IOV $IOV_number found. Being processed..."

	NEWIOV=True

	cat template_DBReader_cfg.py | sed -e "s@insertRun@$IOV_number@g" -e "s@insertLog@$LOGDESTINATION@g" -e "s@insertDB@$DB@g" -e "s@insertFrontier@$FRONTIER@g" -e "s@insertAccount@$ACCOUNT@g" -e "s@insertTag@$tag@g" -e "s@insertRecord@$RECORD@g" -e "s@insertOutFile@$ROOTFILE@g" -e "s@insertPedestalMon@$MONITOR_PEDESTAL@g" -e "s@insertNoiseMon@$MONITOR_NOISE@g" -e "s@insertQualityMon@$MONITOR_QUALITY@g" -e "s@insertGainMon@$MONITOR_GAIN@g" -e "s@insertCablingMon@$MONITOR_CABLING@g" -e "s@insertLorentzAngleMon@$MONITOR_LA@g" -e "s@insertThresholdMon@$MONITOR_THRESHOLD@g" -e "s@insertMonitorCumulative@$MONITORCUMULATIVE@g" -e "s@insertActiveDetId@$USEACTIVEDETID@g"> DBReader_cfg.py
	if [ "$MONITOR_QUALITY" = "True" ]; then
	    cat >> DBReader_cfg.py  << EOF

process.SiStripQualityESProducer = cms.ESProducer("SiStripQualityESProducer",
   ReduceGranularity = cms.bool(False),
   PrintDebugOutput = cms.bool(False),
   UseEmptyRunInfo = cms.bool(False),
   ListOfRecordToMerge = cms.VPSet(cms.PSet(
   record = cms.string('$RECORD'),
   tag = cms.string('')
   ))
)

process.stat = cms.EDFilter("SiStripQualityStatistics",
    TkMapFileName = cms.untracked.string(''),
    dataLabel = cms.untracked.string('')
)

process.e = cms.EndPath(process.stat)
EOF
	fi

	if [ "$MONITOR_CABLING" = "True" ]; then
	    if [ "$ACCOUNT" = "CMS_COND_21X_STRIP" ]; then # For CMSSW_2_x_y the FedCablingReader is not available!
	    cat >> DBReader_cfg.py  << EOF

process.SiStripQualityESProducer = cms.ESProducer("SiStripQualityESProducer",
   ReduceGranularity = cms.bool(False),
   PrintDebugOutput = cms.bool(False),
   UseEmptyRunInfo = cms.bool(False),
   ListOfRecordToMerge = cms.VPSet(cms.PSet(
   record = cms.string('$RECORDFORQUALITY'),
   tag = cms.string('')
   ))
)

process.sistripconn = cms.ESProducer("SiStripConnectivity")

process.stat = cms.EDFilter("SiStripQualityStatistics",
    TkMapFileName = cms.untracked.string(''),
    dataLabel = cms.untracked.string('')
)

process.e = cms.EndPath(process.stat)
EOF
	    else
	    cat >> DBReader_cfg.py  << EOF

process.SiStripQualityESProducer = cms.ESProducer("SiStripQualityESProducer",
   ReduceGranularity = cms.bool(False),
   PrintDebugOutput = cms.bool(False),
   UseEmptyRunInfo = cms.bool(False),
   ListOfRecordToMerge = cms.VPSet(cms.PSet(
   record = cms.string('$RECORDFORQUALITY'),
   tag = cms.string('')
   ))
)

process.sistripconn = cms.ESProducer("SiStripConnectivity")

process.stat = cms.EDFilter("SiStripQualityStatistics",
    TkMapFileName = cms.untracked.string(''),
    dataLabel = cms.untracked.string('')
)

process.reader = cms.EDFilter("SiStripFedCablingReader")

process.e = cms.EndPath(process.stat*process.reader)
EOF
	    fi
	fi

	echo "Executing cmsRun. Stay tuned ..."

	cmsRun DBReader_cfg.py

	echo "cmsRun finished. Now moving the files to the corresponding directories ..."

	if [ "$NEWTAG" = "True" ] && [ "$CFGISSAVED" = "False" ]; then
	    cp DBReader_cfg.py $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/cfg/${tag}_cfg.py
	    CFGISSAVED=True
	fi

	mv $ROOTFILE $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/rootfiles;

	if [ "$MONITOR_QUALITY" = "True" ]; then
	    cat $LOGDESTINATION.log | awk 'BEGIN{doprint=0}{if(match($0,"New IOV")!=0) doprint=1;if(match($0,"%MSG")!=0) {doprint=0;} if(doprint==1) print $0}' > QualityInfo_Run${IOV_number}.txt
	    mv QualityInfo_Run${IOV_number}.txt $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/QualityLog/

	    rm $LOGDESTINATION.log
	fi

	if [ "$MONITOR_CABLING" = "True" ]; then
	    if [ "$ACCOUNT" != "CMS_COND_21X_STRIP" ]; then
		cat $LOGDESTINATION.log | awk 'BEGIN{doprint=0}{if(match($0,"beginRun")!=0) doprint=1;if(match($0,"%MSG")!=0) {doprint=0;} if(doprint==1) print $0}' > CablingInfo_Run${IOV_number}.txt
		mv CablingInfo_Run${IOV_number}.txt $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/CablingLog/
	    fi

	    cat $LOGDESTINATION.log | awk 'BEGIN{doprint=0}{if(match($0,"New IOV")!=0) doprint=1;if(match($0,"%MSG")!=0) {doprint=0;} if(doprint==1) print $0}' > QualityInfoFromCabling_Run${IOV_number}.txt
	    mv QualityInfoFromCabling_Run${IOV_number}.txt $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/CablingLog/

	    rm $LOGDESTINATION.log
	fi

	for Plot in `ls *.png | grep TIB`; do
	    PNGNAME=`echo ${Plot#*_*_*_*_*_} | gawk -F . '{print $1}'`
	    LAYER=`echo ${PNGNAME#*_*_} | gawk -F _ '{print $1}'`
	    if [ "$MONITORCUMULATIVE" = "True" ] || [ "$CREATETRENDS" = "True" ]; then
		if [ `echo $Plot | grep Cumulative` ]; then
		    mv $Plot $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/TIB/Layer$LAYER/Cumulative/${PNGNAME}__Run${IOV_number}.png;
		else
		    mv $Plot $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/TIB/Layer$LAYER/Profile/${PNGNAME}__Run${IOV_number}.png;
		fi
	    else
		mv $Plot $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/TIB/Layer$LAYER/${PNGNAME}__Run${IOV_number}.png;
	    fi
	done;

	for Plot in `ls *.png | grep TOB`; do
	    PNGNAME=`echo ${Plot#*_*_*_*_*_} | gawk -F . '{print $1}'`
	    LAYER=`echo ${PNGNAME#*_*_} | gawk -F _ '{print $1}'`
	    if [ "$MONITORCUMULATIVE" = "True" ] || [ "$CREATETRENDS" = "True" ]; then
		if [ `echo $Plot | grep Cumulative` ]; then
		    mv $Plot $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/TOB/Layer$LAYER/Cumulative/${PNGNAME}__Run${IOV_number}.png;
		else
		    mv $Plot $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/TOB/Layer$LAYER/Profile/${PNGNAME}__Run${IOV_number}.png;
		fi
	    else
		mv $Plot $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/TOB/Layer$LAYER/${PNGNAME}__Run${IOV_number}.png;
	    fi
	done;

	for Plot in `ls *.png | grep TID`; do
	    PNGNAME=`echo ${Plot#*_*_*_*_*_} | gawk -F . '{print $1}'`
	    SIDE=`echo ${PNGNAME#*_*_} | gawk -F _ '{print $1}'`
	    DISK=`echo ${PNGNAME#*_*_*_*_*_*_} | gawk -F _ '{print $1}'`
	    if [ "$MONITORCUMULATIVE" = "True" ] || [ "$CREATETRENDS" = "True" ]; then
		if [ `echo $Plot | grep Cumulative` ]; then
		    mv $Plot $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/TID/Side$SIDE/Disk$DISK/Cumulative/${PNGNAME}__Run${IOV_number}.png;
		else
		    mv $Plot $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/TID/Side$SIDE/Disk$DISK/Profile/${PNGNAME}__Run${IOV_number}.png;
		fi
	    else
		mv $Plot $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/TID/Side$SIDE/Disk$DISK/${PNGNAME}__Run${IOV_number}.png;
	    fi
	done;

	for Plot in `ls *.png | grep TEC`; do
	    PNGNAME=`echo ${Plot#*_*_*_*_*_} | gawk -F . '{print $1}'`
	    SIDE=`echo ${PNGNAME#*_*_} | gawk -F _ '{print $1}'`
	    DISK=`echo ${PNGNAME#*_*_*_*_*_*_} | gawk -F _ '{print $1}'`
	    if [ "$MONITORCUMULATIVE" = "True" ] || [ "$CREATETRENDS" = "True" ]; then
		if [ `echo $Plot | grep Cumulative` ]; then
		    mv $Plot $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/TEC/Side$SIDE/Disk$DISK/Cumulative/${PNGNAME}__Run${IOV_number}.png;
		else
		    mv $Plot $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/TEC/Side$SIDE/Disk$DISK/Profile/${PNGNAME}__Run${IOV_number}.png;
		fi
	    else
		mv $Plot $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/TEC/Side$SIDE/Disk$DISK/${PNGNAME}__Run${IOV_number}.png;
	    fi
	done;

	for Plot in `ls *.png | grep TkMap`; do
	    #PNGNAME=`echo $Plot | gawk -F . '{print $1}'`
	    mv $Plot $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/TrackerMap/$Plot;
	done;

	for Plot in `ls *.png | grep Bad`; do
	    PNGNAME=`echo ${Plot#*_} | gawk -F . '{print $1}'`
	    if      [ `echo $PNGNAME | grep Apvs` ]; then
		mv $Plot $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/Summary/BadAPVs/${PNGNAME}__Run${IOV_number}.png;
	    else if [ `echo $PNGNAME | grep Fibers` ]; then
		mv $Plot $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/Summary/BadFibers/${PNGNAME}__Run${IOV_number}.png;
	    else if [ `echo $PNGNAME | grep Modules` ]; then
		mv $Plot $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/Summary/BadModules/${PNGNAME}__Run${IOV_number}.png;
	    else if [ `echo $PNGNAME | grep Strips` ]; then
		mv $Plot $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/Summary/BadStrips/${PNGNAME}__Run${IOV_number}.png;
	    fi
	    fi
	    fi
	    fi
	done;

	for Plot in `ls *.png | grep Cabling`; do
	    PNGNAME=`echo ${Plot#*_} | gawk -F . '{print $1}'`
	    mv $Plot $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/Summary/${PNGNAME}__Run${IOV_number}.png;
	done;
   
	#cd $WORKDIR;

    done;

    # Run the Trends and Publish all histograms on a web page
    if [ "$NEWTAG" = "True" ] || [ "$NEWIOV" = "True" ]; then

	if [ "$CREATETRENDS" = "True" ]; then
	    echo "Creating the Trend Plots ..."

	    ./getOfflineDQMData.sh $DB $ACCOUNT $TAGSUBDIR $tag

	    for i in {1..4}; do
		for Plot in `ls *.png | grep TIBLayer$i`; do
		    mv $Plot $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/TIB/Layer$i/Trends;
		done
	    done

	    for i in {1..6}; do
		for Plot in `ls *.png | grep TOBLayer$i`; do
		    mv $Plot $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/TOB/Layer$i/Trends;
		done
	    done

	    for i in {1..3}; do
		for Plot in `ls *.png | grep TID-Disk$i`; do
		    mv $Plot $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/TID/Side1/Disk$i/Trends;
		done
		for Plot in `ls *.png | grep TID+Disk$i`; do
		    mv $Plot $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/TID/Side2/Disk$i/Trends;
		done
	    done

	    for i in {1..9}; do
		for Plot in `ls *.png | grep TEC-Disk$i`; do
		    mv $Plot $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/TEC/Side1/Disk$i/Trends;
		done
		for Plot in `ls *.png | grep TEC+Disk$i`; do
		    mv $Plot $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/TEC/Side2/Disk$i/Trends;
		done
	    done
	    
	    for Plot in `ls *.png | grep TIB`; do
		mv $Plot $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/TIB/Trends;
	    done

	    for Plot in `ls *.png | grep TOB`; do
		mv $Plot $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/TOB/Trends;
	    done

	    for Plot in `ls *.png | grep TID-`; do
		mv $Plot $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/TID/Side1/Trends;
	    done

	    for Plot in `ls *.png | grep TID+`; do
		mv $Plot $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/TID/Side2/Trends;
	    done

	    for Plot in `ls *.png | grep TEC-`; do
		mv $Plot $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/TEC/Side1/Trends;
	    done

	    for Plot in `ls *.png | grep TEC+`; do
		mv $Plot $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/TEC/Side2/Trends;
	    done

	    for Plot in `ls *.png | grep Tracker`; do
		mv $Plot $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/Trends;
	    done
	fi

	mv TrackerSummary.root $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/rootfiles;
	rm -f TrackerPlots.root;
	rm -f makePlots_cc.d makePlots_cc.so;
	rm -f makeTKTrend_cc.d makeTKTrend_cc.so;

	echo "Publishing the new tag $tag (or the new IOV) on the web ..."

	for i in {1..4}; do
	    cd /afs/cern.ch/cms/tracker/sistrcalib/WWW;
	    cat template_index_header.html | sed -e "s@insertPageName@$tag --- TIB Layer $i --- Summary Report@g" > index_new.html
	    if [ "$MONITORCUMULATIVE" = "True" ] || [ "$CREATETRENDS" = "True" ]; then
		cd $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/TIB/Layer$i/Profile;
		CreateIndex

		if [ "$MONITORCUMULATIVE" = "True" ]; then
		    cd $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/TIB/Layer$i/Cumulative;
		    CreateIndex
		fi

		if [ "$CREATETRENDS" = "True" ]; then
		    cd $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/TIB/Layer$i/Trends;
		    CreateIndex
		fi
	    else
		cd $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/TIB/Layer$i;
		CreateIndex
	    fi
	done

	for i in {1..6}; do
	    cd /afs/cern.ch/cms/tracker/sistrcalib/WWW;
	    cat template_index_header.html | sed -e "s@insertPageName@$tag --- TOB Layer $i --- Summary Report@g" > index_new.html
	    if [ "$MONITORCUMULATIVE" = "True" ] || [ "$CREATETRENDS" = "True" ]; then
		cd $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/TOB/Layer$i/Profile;
		CreateIndex

		if [ "$MONITORCUMULATIVE" = "True" ]; then
		    cd $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/TOB/Layer$i/Cumulative;
		    CreateIndex
		fi

		if [ "$CREATETRENDS" = "True" ]; then
		    cd $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/TOB/Layer$i/Trends;
		    CreateIndex
		fi
	    else
		cd $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/TOB/Layer$i;
		CreateIndex
	    fi
	done

	for i in {1..2}; do
	    for j in {1..3}; do
		cd /afs/cern.ch/cms/tracker/sistrcalib/WWW;
		cat template_index_header.html | sed -e "s@insertPageName@$tag --- TID Side $i Disk $j --- Summary Report@g" > index_new.html
		if [ "$MONITORCUMULATIVE" = "True" ] || [ "$CREATETRENDS" = "True" ]; then
		    cd $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/TID/Side$i/Disk$j/Profile;
		    CreateIndex

		    if [ "$MONITORCUMULATIVE" = "True" ]; then
			cd $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/TID/Side$i/Disk$j/Cumulative;
			CreateIndex
		    fi

		    if [ "$CREATETRENDS" = "True" ]; then
			cd $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/TID/Side$i/Disk$j/Trends;
			CreateIndex
		    fi
		else
		    cd $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/TID/Side$i/Disk$j;
		    CreateIndex
		fi
	    done
	done

	for i in {1..2}; do
	    for j in {1..9}; do
		cd /afs/cern.ch/cms/tracker/sistrcalib/WWW;
		cat template_index_header.html | sed -e "s@insertPageName@$tag --- TEC Side $i Disk $j --- Summary Report@g" > index_new.html
		if [ "$MONITORCUMULATIVE" = "True" ] || [ "$CREATETRENDS" = "True" ]; then
		    cd $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/TEC/Side$i/Disk$j/Profile;
		    CreateIndex

		    if [ "$MONITORCUMULATIVE" = "True" ]; then
			cd $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/TEC/Side$i/Disk$j/Cumulative;
			CreateIndex
		    fi

		    if [ "$CREATETRENDS" = "True" ]; then
			cd $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/TEC/Side$i/Disk$j/Trends;
			CreateIndex
		    fi
		else
		    cd $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/TEC/Side$i/Disk$j;
		    CreateIndex
		fi
	    done
	done

	if [ "$CREATETRENDS" = "True" ]; then
	    cd /afs/cern.ch/cms/tracker/sistrcalib/WWW;
	    cat template_index_header.html | sed -e "s@insertPageName@$tag --- Full Strip Tracker --- Trend Plots@g" > index_new.html
	    cd $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/Trends;
	    CreateIndex

	    cd /afs/cern.ch/cms/tracker/sistrcalib/WWW;
	    cat template_index_header.html | sed -e "s@insertPageName@$tag --- TIB --- Trend Plots@g" > index_new.html
	    cd $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/TIB/Trends;
	    CreateIndex

	    cd /afs/cern.ch/cms/tracker/sistrcalib/WWW;
	    cat template_index_header.html | sed -e "s@insertPageName@$tag --- TOB --- Trend Plots@g" > index_new.html
	    cd $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/TOB/Trends;
	    CreateIndex

	    for i in {1..2}; do
		cd /afs/cern.ch/cms/tracker/sistrcalib/WWW;
		cat template_index_header.html | sed -e "s@insertPageName@$tag --- TID Side $i --- Trend Plots@g" > index_new.html
		cd $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/TID/Side$i/Trends;
		CreateIndex

		cd /afs/cern.ch/cms/tracker/sistrcalib/WWW;
		cat template_index_header.html | sed -e "s@insertPageName@$tag --- TEC Side $i --- Trend Plots@g" > index_new.html
		cd $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/TEC/Side$i/Trends;
		CreateIndex
	    done
	fi

	if [ "$MONITOR_QUALITY" = "True" ]; then
	    cd /afs/cern.ch/cms/tracker/sistrcalib/WWW;
	    cat template_index_header.html | sed -e "s@insertPageName@$tag --- Bad APVs --- Summary Report@g" > index_new.html
	    cd $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/Summary/BadAPVs;
	    CreateIndex

	    cd /afs/cern.ch/cms/tracker/sistrcalib/WWW;
	    cat template_index_header.html | sed -e "s@insertPageName@$tag --- Bad Fibers --- Summary Report@g" > index_new.html
	    cd $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/Summary/BadFibers;
	    CreateIndex
	    
	    cd /afs/cern.ch/cms/tracker/sistrcalib/WWW;
	    cat template_index_header.html | sed -e "s@insertPageName@$tag --- Bad Modules --- Summary Report@g" > index_new.html
	    cd $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/Summary/BadModules;
	    CreateIndex

	    cd /afs/cern.ch/cms/tracker/sistrcalib/WWW;
	    cat template_index_header.html | sed -e "s@insertPageName@$tag --- Bad Strips --- Summary Report@g" > index_new.html
	    cd $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/Summary/BadStrips;
	    CreateIndex
		
	fi
    
	if [ "$MONITOR_CABLING" = "True" ]; then
	    cd /afs/cern.ch/cms/tracker/sistrcalib/WWW;
	    cat template_index_header.html | sed -e "s@insertPageName@$tag --- Summary Report@g" > index_new.html
	    cd $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/Summary/;
	    CreateIndex
		
	fi

	if [ -d "$STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/TrackerMap" ]; then
	    cd /afs/cern.ch/cms/tracker/sistrcalib/WWW;
	    cat template_index_header.html | sed -e "s@insertPageName@$tag --- Tracker Maps for all IOVs ---@g" > index_new.html
	    cd $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/TrackerMap;
	    CreateIndex

	fi

    fi

    cd $WORKDIR;

done;
