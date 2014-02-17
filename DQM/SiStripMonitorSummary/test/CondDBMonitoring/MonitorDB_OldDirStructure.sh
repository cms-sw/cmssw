#!/bin/bash

#export PATH=$PATH:/afs/cern.ch/cms/sw/common/
#export FRONTIER_FORCERELOAD=long # This should not be used anymore!!!

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
#Example: FRONTIER=frontier://cmsfrontier.cern.ch:8000/FrontierProd
FRONTIER=$4
DBTAGCOLLECTION=DBTagsIn_${DB}_${ACCOUNT}.txt
GLOBALTAGCOLLECTION=GlobalTagsForDBTag.txt
DBTAGDIR=DBTagCollection
GLOBALTAGDIR=GlobalTags
STORAGEPATH=/afs/cern.ch/cms/tracker/sistrcalib/WWW/CondDBMonitoring
WORKDIR=$PWD

#eval `scramv1 runtime -sh`

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

if [ ! -d "$STORAGEPATH/$DB/$ACCOUNT/$GLOBALTAGDIR" ]; then 
    echo "Creating directory $STORAGEPATH/$DB/$ACCOUNT/$GLOBALTAGDIR"
    mkdir $STORAGEPATH/$DB/$ACCOUNT/$GLOBALTAGDIR; 
fi

# Access of all Tags uploaded to the given DB account
#cmscond_list_iov -c oracle://$DB/$ACCOUNT -P /afs/cern.ch/cms/DB/conddb > $DBTAGCOLLECTION # Access via oracle
cmscond_list_iov -c $FRONTIER/$ACCOUNT -P /afs/cern.ch/cms/DB/conddb > $DBTAGCOLLECTION # Access via Frontier

# Loop on all DB Tags
for tag in `cat $DBTAGCOLLECTION`; do

    echo "Processing DB-Tag $tag";

    # Discover which kind of tag is processed
    MONITOR_NOISE=False
    MONITOR_PEDESTAL=False
    MONITOR_GAIN=False
    MONITOR_QUALITY=False
    MONITOR_CABLING=False
    MONITOR_LA=False
    MONITOR_THRESHOLD=False

    if      [ `echo $tag | grep "Noise" | wc -w` -gt 0 ]; then
	MONITOR_NOISE=True
	USEACTIVEDETID=True
	RECORD=SiStripNoisesRcd
	TAGSUBDIR=SiStripNoise
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
    else if [ `echo $tag | grep "Cabling" | wc -w` -gt 0 ]; then
	MONITOR_CABLING=True
	USEACTIVEDETID=True
	RECORD=SiStripFedCablingRcd
	TAGSUBDIR=SiStripFedCabling
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

    # Creation of DB-Tag directory if not existing yet
    if [ ! -d "$STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR" ]; then 
	echo "Creating directory $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR"
	mkdir $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR; 
    fi

    if [ ! -d "$STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag" ]; then 
	echo "Creating directory $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag"
	mkdir $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag;
	mkdir $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/RelatedGlobalTags;
    fi

    # Access of all Global Tags for the given DB Tag
    if [ -f globaltag_tmp.txt ]; then
	rm globaltag_tmp.txt;
    fi

    if [ -f $GLOBALTAGCOLLECTION ]; then
	rm $GLOBALTAGCOLLECTION;
    fi

    if [ `echo $DB | grep "prep" | wc -w` -gt 0 ]; then
	for globaltag in `cmscond_tagintrees -c sqlite_file:$WORKDIR/CMSSW_3_1_0_pre6/src/CondCore/TagCollection/data/GlobalTag.db -P /afs/cern.ch/cms/DB/conddb -t $tag | grep Trees`; do # Access via Frontier

	    if [ "$globaltag" != "#" ] && [ "$globaltag" != "Trees" ]; then
		echo $globaltag >> globaltag_tmp.txt;
		cat globaltag_tmp.txt | sed -e "s@\[@@g" -e "s@\]@@g" -e "s@'@@g" -e "s@,@@g" > $GLOBALTAGCOLLECTION
	    fi

	done;

    else
	for globaltag in `cmscond_tagintrees -c $FRONTIER/$GTACCOUNT -P /afs/cern.ch/cms/DB/conddb -t $tag | grep Trees`; do # Access via Frontier

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
<a href="http://cmstac05.cern.ch/CondDBMonitoring/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag">http://cmstac05.cern.ch/CondDBMonitoring/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag</a>
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
<a href="http://cmstac05.cern.ch/CondDBMonitoring/$DB/$ACCOUNT/$GLOBALTAGDIR/$globaltag">http://cmstac05.cern.ch/CondDBMonitoring/$DB/$ACCOUNT/$GLOBALTAGDIR/$globaltag</a>
</body>
</html>
EOF
	    #ln -s $STORAGEPATH/$DB/$ACCOUNT/$GLOBALTAGDIR/$globaltag $globaltag;
	    cd $WORKDIR;

	done;

    fi

    # Get the list of IoVs for the given DB-Tag
    #cmscond_list_iov -c oracle://$DB/$ACCOUNT -P /afs/cern.ch/cms/DB/conddb -t $tag | awk '{if(NR>4) print "Run_In "$1 " Run_End " $2}' > list_Iov.txt  # Access via oracle
    cmscond_list_iov -c $FRONTIER/$ACCOUNT -P /afs/cern.ch/cms/DB/conddb -t $tag | awk '{if(NR>4) print "Run_In "$1 " Run_End " $2}' > list_Iov.txt # Access via Frontier

    # Access DB for the given DB-Tag and dump values in a root-file and histograms in .png if not existing yet
    echo "Now the values are retrieved from the DB..."
    
    if [ ! -d "$STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/rootfiles" ]; then
	mkdir $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/rootfiles;
	mkdir $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots;
	mkdir $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/TIB;
	mkdir $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/TOB;
	mkdir $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/TID;
	mkdir $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/TEC;
	mkdir $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/TrackerMap;

	if [ "$MONITOR_QUALITY" = "True" ] || [ "$MONITOR_CABLING" = "True" ]; then
	mkdir $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/Summary
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
	
	if [ -d $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/TIB/$IOV_number ]; then # Skip IOVs already processed. Take only new ones.
	    continue
	fi

	echo "New IOV $IOV_number found. Being processed..."

	mkdir $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/TIB/$IOV_number;
	mkdir $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/TOB/$IOV_number;
	mkdir $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/TID/$IOV_number;
	mkdir $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/TEC/$IOV_number;

	if [ "$RECORD" = "Unknown" ]; then
	    echo "Unknown strip tag. Processing skipped!"
	    continue
	fi

	if [ "$MONITOR_QUALITY" = "True" ] || [ "$MONITOR_CABLING" = "True" ]; then
	    mkdir $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/Summary/$IOV_number;
	fi

	ROOTFILE="${tag}_Run_${IOV_number}.root"

	cat template_DBReader_cfg.py | sed -e "s@insertRun@$IOV_number@g" -e "s@insertDB@$DB@g" -e "s@insertFrontier@$FRONTIER@g" -e "s@insertAccount@$ACCOUNT@g" -e "s@insertTag@$tag@g" -e "s@insertRecord@$RECORD@g" -e "s@insertOutFile@$ROOTFILE@g" -e "s@insertPedestalMon@$MONITOR_PEDESTAL@g" -e "s@insertNoiseMon@$MONITOR_NOISE@g" -e "s@insertQualityMon@$MONITOR_QUALITY@g" -e "s@insertGainMon@$MONITOR_GAIN@g" -e "s@insertCablingMon@$MONITOR_CABLING@g" -e "s@insertLorentzAngleMon@$MONITOR_LA@g" -e "s@insertThresholdMon@$MONITOR_THRESHOLD@g" -e "s@insertActiveDetId@$USEACTIVEDETID@g"> DBReader_cfg.py
	if [ "$MONITOR_QUALITY" = "True" ]; then
	    cat >> DBReader_cfg.py  << EOF

process.SiStripQualityESProducer = cms.ESProducer("SiStripQualityESProducer",
   ReduceGranularity = cms.bool(False),
   ListOfRecordToMerge = cms.VPSet(cms.PSet(
   record = cms.string('$RECORD'),
   tag = cms.string('')
   ))
)
EOF
	fi

	if [ "$MONITOR_CABLING" = "True" ]; then
	    cat >> DBReader_cfg.py  << EOF

process.sistripconn = cms.ESProducer("SiStripConnectivity")
EOF
	fi

	cmsRun DBReader_cfg.py

	mv $ROOTFILE $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/rootfiles;

	for Plot in `ls *.png | grep TIB`; do
	    PNGNAME=`echo ${Plot#*_*_*_*_*_} | gawk -F . '{print $1}'`
	    mv $Plot $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/TIB/$IOV_number/${PNGNAME}__Run${IOV_number}.png;
	    
	done;

	for Plot in `ls *.png | grep TOB`; do
	    PNGNAME=`echo ${Plot#*_*_*_*_*_} | gawk -F . '{print $1}'`
	    mv $Plot $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/TOB/$IOV_number/${PNGNAME}__Run${IOV_number}.png;

	done;

	for Plot in `ls *.png | grep TID`; do
	    PNGNAME=`echo ${Plot#*_*_*_*_*_} | gawk -F . '{print $1}'`
	    mv $Plot $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/TID/$IOV_number/${PNGNAME}__Run${IOV_number}.png;

	done;

	for Plot in `ls *.png | grep TEC`; do
	    PNGNAME=`echo ${Plot#*_*_*_*_*_} | gawk -F . '{print $1}'`
	    mv $Plot $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/TEC/$IOV_number/${PNGNAME}__Run${IOV_number}.png;

	done;

	for Plot in `ls *.png | grep Bad`; do
	    PNGNAME=`echo ${Plot#*_} | gawk -F . '{print $1}'`
	    mv $Plot $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/Summary/$IOV_number/${PNGNAME}__Run${IOV_number}.png;

	done;

	for Plot in `ls *.png | grep Cabling`; do
	    PNGNAME=`echo ${Plot#*_} | gawk -F . '{print $1}'`
	    mv $Plot $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/Summary/$IOV_number/${PNGNAME}__Run${IOV_number}.png;

	done;

	for Plot in `ls *.png | grep TkMap`; do
	    #PNGNAME=`echo $Plot | gawk -F . '{print $1}'`
	    mv $Plot $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/TrackerMap/$Plot;

	done;

        # Publish the histograms on a web page
	cd /afs/cern.ch/cms/tracker/sistrcalib/WWW;
	cat template_createWPageForDBMonitoring.pl | sed -e "s@insertPageName@$tag --- TIB --- Summary Report --- IOV $IOV_number@g" > createWPageForDBMonitoring.pl
	cd $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/TIB/$IOV_number;
	perl /afs/cern.ch/cms/tracker/sistrcalib/WWW/createWPageForDBMonitoring.pl

	cd /afs/cern.ch/cms/tracker/sistrcalib/WWW;
	cat template_createWPageForDBMonitoring.pl | sed -e "s@insertPageName@$tag --- TOB --- Summary Report --- IOV $IOV_number@g" > createWPageForDBMonitoring.pl
	cd $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/TOB/$IOV_number;
	perl /afs/cern.ch/cms/tracker/sistrcalib/WWW/createWPageForDBMonitoring.pl

	cd /afs/cern.ch/cms/tracker/sistrcalib/WWW;
	cat template_createWPageForDBMonitoring.pl | sed -e "s@insertPageName@$tag --- TID --- Summary Report --- IOV $IOV_number@g" > createWPageForDBMonitoring.pl
	cd $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/TID/$IOV_number;
	perl /afs/cern.ch/cms/tracker/sistrcalib/WWW/createWPageForDBMonitoring.pl

	cd /afs/cern.ch/cms/tracker/sistrcalib/WWW;
	cat template_createWPageForDBMonitoring.pl | sed -e "s@insertPageName@$tag --- TEC --- Summary Report --- IOV $IOV_number@g" > createWPageForDBMonitoring.pl
	cd $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/TEC/$IOV_number;
	perl /afs/cern.ch/cms/tracker/sistrcalib/WWW/createWPageForDBMonitoring.pl

	if [ "$MONITOR_QUALITY" = "True" ] || [ "$MONITOR_CABLING" = "True" ]; then
	    cd /afs/cern.ch/cms/tracker/sistrcalib/WWW;
	    cat template_createWPageForDBMonitoring.pl | sed -e "s@insertPageName@$tag --- Summary Report --- IOV $IOV_number@g" > createWPageForDBMonitoring.pl
	    cd $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/Summary/$IOV_number;
	    perl /afs/cern.ch/cms/tracker/sistrcalib/WWW/createWPageForDBMonitoring.pl
		
	fi
	    
	cd $WORKDIR;

    done;

    if [ -d "$STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/TrackerMap" ]; then
	cd /afs/cern.ch/cms/tracker/sistrcalib/WWW;
	cat template_createWPageForDBMonitoring.pl | sed -e "s@insertPageName@$tag --- Tracker Maps for all IOVs---@g" > createWPageForDBMonitoring.pl
	cd $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/TrackerMap;
	perl /afs/cern.ch/cms/tracker/sistrcalib/WWW/createWPageForDBMonitoring.pl

	cd $WORKDIR;
    fi

done;
