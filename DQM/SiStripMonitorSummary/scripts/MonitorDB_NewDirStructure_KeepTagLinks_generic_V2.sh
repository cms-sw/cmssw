#!/bin/bash

# needed to allow the loop on *.png without using "*.png" as value
shopt -s nullglob
date

if [ $# -ne 3 ]; then
    afstokenchecker.sh "You have to provide a <tag_search_string>, a <DB> and a DB name for the connection string!!!"
    exit
fi

afstokenchecker.sh "Starting execution of MonitorDB_NewDirStructure_KeepTagLinks_generic_V2 $1 $2 $3"

#Example: SEARCHSTRING=SiStrip
SEARCHSTRING=$1
#Example: DB=cms_orcoff_prod
DB=$2
ACCOUNT=CMS_CONDITIONS #to be checked if it is ok
FRONTIER=$3 #to be fixed for dev?
DBTAGCOLLECTION=DBTagsIn_${DB}.txt
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
    afstokenchecker.sh "Creating directory $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR"
    mkdir $STORAGEPATH/$DB/$ACCOUNT; 
fi

if [ ! -d "$STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR" ]; then 
    afstokenchecker.sh "Creating directory $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR"
    mkdir $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR; 
fi

# Creation of Global Tag directory from scratch to have always up-to-date information (and no old links that are obsolete)
if [ ! -d "$STORAGEPATH/$DB/$GLOBALTAGDIR" ]; then 
    afstokenchecker.sh "Creating directory $STORAGEPATH/$DB/$GLOBALTAGDIR"
    mkdir $STORAGEPATH/$DB/$GLOBALTAGDIR; 
fi

# Access of all SiStrip Tags uploaded to the given DB account
conddb --db $DB --nocolors listTags | grep --regexp=^$SEARCHSTRING | awk '{if(match($1,"V0")!=0 || match($1,"/")!=0) {} else {print $1}}' > $DBTAGCOLLECTION

# Loop on all DB Tags
for tag in `cat $DBTAGCOLLECTION`; do

    afstokenchecker.sh "Processing DB-Tag $tag";

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
    MONITOR_BP=False
    MONITOR_THRESHOLD=False
    MONITOR_LATENCY=False
    MONITOR_SHIFTANDCROSSTALK=False
    MONITOR_APVPHASEOFFSETS=False
    MONITOR_ALCARECOTRIGGERBITS=False

    LOGDESTINATION=cout
    QUALITYLOGDEST=QualityInfo
    CABLINGLOGDEST=CablingInfo
    CONDLOGDEST=Dummy

    RECORDFORQUALITY=Dummy

    MONITORCUMULATIVE=False
    USEACTIVEDETID=False
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
	QUALITYLOGDEST=QualityInfo
	CREATETRENDS=True
    else if [ `echo $tag | grep "Cabling" | wc -w` -gt 0 ]; then
	MONITOR_CABLING=True
	USEACTIVEDETID=True
	RECORD=SiStripFedCablingRcd
	RECORDFORQUALITY=SiStripDetCablingRcd
	TAGSUBDIR=SiStripFedCabling
	LOGDESTINATION=Reader
	QUALITYLOGDEST=QualityInfoFromCabling
	CABLINGLOGDEST=CablingInfo
	CREATETRENDS=True
    else if [ `echo $tag | grep "Lorentz" | wc -w` -gt 0 ]; then
	MONITOR_LA=True
	USEACTIVEDETID=False
	RECORD=SiStripLorentzAngleRcd
	TAGSUBDIR=SiStripLorentzAngle
    else if [ `echo $tag | grep "BackPlane" | wc -w` -gt 0 ]; then
	MONITOR_BP=True
	USEACTIVEDETID=False
	RECORD=SiStripBackPlaneCorrectionRcd
	TAGSUBDIR=SiStripBackPlaneCorrection
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
    else if [ `echo $tag | grep "Latency" | wc -w` -gt 0 ]; then
	MONITOR_LATENCY=True
	RECORD=SiStripLatencyRcd
	TAGSUBDIR=SiStripLatency
	LOGDESTINATION=Reader
	CONDLOGDEST=LatencyInfo
    else if [ `echo $tag | grep "Shift" | wc -w` -gt 0 ]; then
	MONITOR_SHIFTANDCROSSTALK=True
	RECORD=SiStripConfObjectRcd
	TAGSUBDIR=SiStripShiftAndCrosstalk
	LOGDESTINATION=Reader
	CONDLOGDEST=ShiftAndCrosstalkInfo
    else if [ `echo $tag | grep "APVPhaseOffsets" | wc -w` -gt 0 ]; then
	MONITOR_APVPHASEOFFSETS=True
	RECORD=SiStripConfObjectRcd
	TAGSUBDIR=SiStripAPVPhaseOffsets
	LOGDESTINATION=Reader
	CONDLOGDEST=APVPhaseOffsetsInfo
    else if [ `echo $tag | grep "AlCaRecoTriggerBits" | wc -w` -gt 0 ]; then
	MONITOR_ALCARECOTRIGGERBITS=True
	RECORD=AlCaRecoTriggerBitsRcd
	TAGSUBDIR=SiStripDQM
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
    fi
    fi
    fi
    fi
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

    if [ "$RECORD" = "Unknown" ]; then
	afstokenchecker.sh "Unknown strip tag. Processing skipped!"
	continue
    fi

    # Get the list of IoVs for the given DB-Tag
    afstokenchecker.sh "Getting the list of IOVs for the given DB tag..."
    conddb --db $DB --nocolors list -L 5000 $tag | awk '{if(match($1,"[a-z]")!=0 || match($1,"-")!=0) {} else {print $1}}' > list_Iov.txt

    # Access DB for the given DB-Tag and dump values in a root-file and histograms in .png if not existing yet
    afstokenchecker.sh "Now the values are retrieved from the DB..."
    
    if [ ! -d "$STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/rootfiles" ]; then
	mkdir $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/rootfiles;
	mkdir $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/Documentation;
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

	if [ "$MONITOR_LATENCY" = "True" ]; then
	    mkdir $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/LatencyLog
	fi

	if [ "$MONITOR_SHIFTANDCROSSTALK" = "True" ]; then
	    mkdir $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/ShiftAndCrosstalkLog
	fi

	if [ "$MONITOR_APVPHASEOFFSETS" = "True" ]; then
	    mkdir $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/APVPhaseOffsetsLog
	fi

	if [ "$MONITOR_ALCARECOTRIGGERBITS" = "True" ]; then
	    mkdir $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/AlCaRecoTriggerBitsLog
	fi

    fi

    if [ `echo *.png | wc -w` -gt 0 ]; then
	rm *.png;
    fi


    # Process each IOV of the given DB-Tag seperately
    for IOV_number in `cat list_Iov.txt`; do

	if [ "$IOV_number" = "Total" ] || [ $IOV_number -gt 100000000 ]; then  # do not loop on time-based IOVs 
	    continue
	fi
	
	ROOTFILE="${tag}_Run_${IOV_number}.root"

	if [ -f $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/rootfiles/$ROOTFILE ]; then # Skip IOVs already processed. Take only new ones.
	    continue
	fi

	if [ "$MONITOR_LATENCY" = "True" ] && [ -f $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/LatencyLog/LatencyInfo_Run${IOV_number}.txt ]; then # Skip IOVs already processed. Take only new ones.
	    continue
	fi

	if [ "$MONITOR_SHIFTANDCROSSTALK" = "True" ] && [ -f $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/ShiftAndCrosstalkLog/ShiftAndCrosstalkInfo_Run${IOV_number}.txt ]; then # Skip IOVs already processed. Take only new ones.
	    continue
	fi

	if [ "$MONITOR_APVPHASEOFFSETS" = "True" ] && [ -f $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/APVPhaseOffsetsLog/APVPhaseOffsetsInfo_Run${IOV_number}.txt ]; then # Skip IOVs already processed. Take only new ones.
	    continue
	fi

	if [ "$MONITOR_ALCARECOTRIGGERBITS" = "True" ] && [ -f $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/AlCaRecoTriggerBitsLog/AlCaRecoTriggerBitsInfo_Run${IOV_number}.txt ]; then # Skip IOVs already processed. Take only new ones.
	    continue
	fi

	afstokenchecker.sh "New IOV $IOV_number found. Being processed..."

	NEWIOV=True

	afstokenchecker.sh "Executing cmsRun. Stay tuned ..."
	CMSRUNCOMMAND="cmsRun ${CMSSW_BASE}/src/DQM/SiStripMonitorSummary/test/DBReader_conddbmonitoring_generic_cfg.py print logDestination=$LOGDESTINATION qualityLogDestination=$QUALITYLOGDEST cablingLogDestination=$CABLINGLOGDEST condLogDestination=$CONDLOGDEST outputRootFile=$ROOTFILE connectionString=frontier://$FRONTIER/$ACCOUNT recordName=$RECORD recordForQualityName=$RECORDFORQUALITY tagName=$tag runNumber=$IOV_number LatencyMon=$MONITOR_LATENCY ALCARecoTriggerBitsMon=$MONITOR_ALCARECOTRIGGERBITS ShiftAndCrosstalkMon=$MONITOR_SHIFTANDCROSSTALK APVPhaseOffsetsMon=$MONITOR_APVPHASEOFFSETS PedestalMon=$MONITOR_PEDESTAL NoiseMon=$MONITOR_NOISE QualityMon=$MONITOR_QUALITY CablingMon=$MONITOR_CABLING GainMon=$MONITOR_GAIN LorentzAngleMon=$MONITOR_LA BackPlaneCorrectionMon=$MONITOR_BP ThresholdMon=$MONITOR_THRESHOLD MonitorCumulative=$MONITORCUMULATIVE ActiveDetId=$USEACTIVEDETID"
	$CMSRUNCOMMAND

	afstokenchecker.sh "cmsRun finished. Now moving the files to the corresponding directories ..."

	cp ${CMSSW_BASE}/src/DQM/SiStripMonitorSummary/test/DBReader_conddbmonitoring_generic_cfg.py DBReader_cfg.py
	cat >> DBReader_cfg.py << EOF 
#
# $CMSRUNCOMMAND
#
EOF

	mv DBReader_cfg.py $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/cfg/${tag}_cfg.py
	CFGISSAVED=True

	if [ "$NEWTAG" = "True" ]; then
	    cat >> ${tag}_documentation << EOF
<html>
<body>
<a href="https://twiki.cern.ch/twiki/bin/view/CMS/StripTrackerDBTagsForCalibrations#${tag}">https://twiki.cern.ch/twiki/bin/view/CMS/StripTrackerDBTagsForCalibrations#${tag}</a>
</body>
</html>
EOF

	    mv ${tag}_documentation $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/Documentation;
	fi

	if [ -f $ROOTFILE ]; then mv $ROOTFILE $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/rootfiles; fi

	if [ "$MONITOR_QUALITY" = "True" ]; then
#	    cat $LOGDESTINATION.log | awk 'BEGIN{doprint=0}{if(match($0,"New IOV")!=0) doprint=1;if(match($0,"%MSG")!=0) {doprint=0;} if(doprint==1) print $0}' > QualityInfo_Run${IOV_number}.txt
	    mv $QUALITYLOGDEST.log QualityInfo_Run${IOV_number}.txt
	    mv QualityInfo_Run${IOV_number}.txt $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/QualityLog/

	fi

	if [ "$MONITOR_CABLING" = "True" ]; then
	    if [ "$ACCOUNT" != "CMS_COND_21X_STRIP" ]; then
#		cat $LOGDESTINATION.log | awk 'BEGIN{doprint=0}{if(match($0,"beginRun")!=0) doprint=1;if(match($0,"%MSG")!=0) {doprint=0;} if(doprint==1) print $0}' > CablingInfo_Run${IOV_number}.txt
		mv $CABLINGLOGDEST.log CablingInfo_Run${IOV_number}.txt
		mv CablingInfo_Run${IOV_number}.txt $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/CablingLog/
	    fi

#	    cat $LOGDESTINATION.log | awk 'BEGIN{doprint=0}{if(match($0,"New IOV")!=0) doprint=1;if(match($0,"%MSG")!=0) {doprint=0;} if(doprint==1) print $0}' > QualityInfoFromCabling_Run${IOV_number}.txt
	    mv $QUALITYLOGDEST.log QualityInfoFromCabling_Run${IOV_number}.txt
	    mv QualityInfoFromCabling_Run${IOV_number}.txt $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/CablingLog/

	fi

	if [ "$MONITOR_LATENCY" = "True" ]; then
#	    cat $LOGDESTINATION.log | awk 'BEGIN{doprint=0}{if(match($0,"PrintSummary")!=0) doprint=1;if(match($0,"PrintDebug")!=0) doprint=1;if(match($0,"%MSG")!=0) {doprint=0;} if(doprint==1) print $0}' > LatencyInfo_Run${IOV_number}.txt
	    mv $CONDLOGDEST.log LatencyInfo_Run${IOV_number}.txt
	    mv LatencyInfo_Run${IOV_number}.txt $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/LatencyLog/

	fi

	if [ "$MONITOR_SHIFTANDCROSSTALK" = "True" ]; then
#	    cat $LOGDESTINATION.log | awk 'BEGIN{doprint=0}{if(match($0,"PrintSummary")!=0) doprint=1;if(match($0,"PrintDebug")!=0) doprint=1;if(match($0,"%MSG")!=0) {doprint=0;} if(doprint==1) print $0}' > ShiftAndCrosstalkInfo_Run${IOV_number}.txt
	    mv $CONDLOGDEST.log ShiftAndCrosstalkInfo_Run${IOV_number}.txt
	    mv ShiftAndCrosstalkInfo_Run${IOV_number}.txt $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/ShiftAndCrosstalkLog/

	fi

	if [ "$MONITOR_APVPHASEOFFSETS" = "True" ]; then
#	    cat $LOGDESTINATION.log | awk 'BEGIN{doprint=0}{if(match($0,"PrintSummary")!=0) doprint=1;if(match($0,"PrintDebug")!=0) doprint=1;if(match($0,"%MSG")!=0) {doprint=0;} if(doprint==1) print $0}' > APVPhaseOffsetsInfo_Run${IOV_number}.txt
	    mv $CONDLOGDEST.log APVPhaseOffsetsInfo_Run${IOV_number}.txt
	    mv APVPhaseOffsetsInfo_Run${IOV_number}.txt $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/APVPhaseOffsetsLog/

	fi

	if [ "$MONITOR_ALCARECOTRIGGERBITS" = "True" ]; then
	    mv AlCaRecoTriggerBitsInfo_Run${IOV_number}.txt $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/AlCaRecoTriggerBitsLog/
	fi

    if [ -f $LOGDESTINATION.log ]; then	rm $LOGDESTINATION.log;  fi
    if [ -f $QUALITYLOGDEST.log ]; then	rm $QUALITYLOGDEST.log;  fi
    if [ -f $CABLINGLOGDEST.log ]; then	rm $CABLINGLOGDEST.log;  fi
    if [ -f $CONDLOGDEST.log ]; then	rm $CONDLOGDEST.log;  fi
	

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
	    afstokenchecker.sh "Creating the Trend Plots ..."

	    getOfflineDQMData.sh $DB $ACCOUNT $TAGSUBDIR $tag $STORAGEPATH

	    afstokenchecker.sh "Moving Trend Plots Using the script moveTrendPlots..."

            moveTrendPlots.sh $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag


	fi

	afstokenchecker.sh "Publishing the new tag $tag (or the new IOV) on the web ..."

	for i in {1..4}; do
	    cd /afs/cern.ch/cms/tracker/sistrcalib/WWW;
	    cat ${CMSSW_BASE}/src/DQM/SiStripMonitorSummary/data/template_index_header.html | sed -e "s@insertPageName@$tag --- TIB Layer $i --- Summary Report@g" > index_new.html
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
	    cat ${CMSSW_BASE}/src/DQM/SiStripMonitorSummary/data/template_index_header.html | sed -e "s@insertPageName@$tag --- TOB Layer $i --- Summary Report@g" > index_new.html
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
		cat ${CMSSW_BASE}/src/DQM/SiStripMonitorSummary/data/template_index_header.html | sed -e "s@insertPageName@$tag --- TID Side $i Disk $j --- Summary Report@g" > index_new.html
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
		cat ${CMSSW_BASE}/src/DQM/SiStripMonitorSummary/data/template_index_header.html | sed -e "s@insertPageName@$tag --- TEC Side $i Disk $j --- Summary Report@g" > index_new.html
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
	    cat ${CMSSW_BASE}/src/DQM/SiStripMonitorSummary/data/template_index_header.html | sed -e "s@insertPageName@$tag --- Full Strip Tracker --- Trend Plots@g" > index_new.html
	    cd $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/Trends;
	    CreateIndex

	    cd /afs/cern.ch/cms/tracker/sistrcalib/WWW;
	    cat ${CMSSW_BASE}/src/DQM/SiStripMonitorSummary/data/template_index_header.html | sed -e "s@insertPageName@$tag --- TIB --- Trend Plots@g" > index_new.html
	    cd $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/TIB/Trends;
	    CreateIndex

	    cd /afs/cern.ch/cms/tracker/sistrcalib/WWW;
	    cat ${CMSSW_BASE}/src/DQM/SiStripMonitorSummary/data/template_index_header.html | sed -e "s@insertPageName@$tag --- TOB --- Trend Plots@g" > index_new.html
	    cd $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/TOB/Trends;
	    CreateIndex

	    for i in {1..2}; do
		cd /afs/cern.ch/cms/tracker/sistrcalib/WWW;
		cat ${CMSSW_BASE}/src/DQM/SiStripMonitorSummary/data/template_index_header.html | sed -e "s@insertPageName@$tag --- TID Side $i --- Trend Plots@g" > index_new.html
		cd $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/TID/Side$i/Trends;
		CreateIndex

		cd /afs/cern.ch/cms/tracker/sistrcalib/WWW;
		cat ${CMSSW_BASE}/src/DQM/SiStripMonitorSummary/data/template_index_header.html | sed -e "s@insertPageName@$tag --- TEC Side $i --- Trend Plots@g" > index_new.html
		cd $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/TEC/Side$i/Trends;
		CreateIndex
	    done
	fi

	if [ "$MONITOR_QUALITY" = "True" ]; then
	    cd /afs/cern.ch/cms/tracker/sistrcalib/WWW;
	    cat ${CMSSW_BASE}/src/DQM/SiStripMonitorSummary/data/template_index_header.html | sed -e "s@insertPageName@$tag --- Bad APVs --- Summary Report@g" > index_new.html
	    cd $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/Summary/BadAPVs;
	    CreateIndex

	    cd /afs/cern.ch/cms/tracker/sistrcalib/WWW;
	    cat ${CMSSW_BASE}/src/DQM/SiStripMonitorSummary/data/template_index_header.html | sed -e "s@insertPageName@$tag --- Bad Fibers --- Summary Report@g" > index_new.html
	    cd $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/Summary/BadFibers;
	    CreateIndex
	    
	    cd /afs/cern.ch/cms/tracker/sistrcalib/WWW;
	    cat ${CMSSW_BASE}/src/DQM/SiStripMonitorSummary/data/template_index_header.html | sed -e "s@insertPageName@$tag --- Bad Modules --- Summary Report@g" > index_new.html
	    cd $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/Summary/BadModules;
	    CreateIndex

	    cd /afs/cern.ch/cms/tracker/sistrcalib/WWW;
	    cat ${CMSSW_BASE}/src/DQM/SiStripMonitorSummary/data/template_index_header.html | sed -e "s@insertPageName@$tag --- Bad Strips --- Summary Report@g" > index_new.html
	    cd $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/Summary/BadStrips;
	    CreateIndex
		
	fi
    
	if [ "$MONITOR_CABLING" = "True" ]; then
	    cd /afs/cern.ch/cms/tracker/sistrcalib/WWW;
	    cat ${CMSSW_BASE}/src/DQM/SiStripMonitorSummary/data/template_index_header.html | sed -e "s@insertPageName@$tag --- Summary Report@g" > index_new.html
	    cd $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/Summary/;
	    CreateIndex
		
	fi

	if [ -d "$STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/TrackerMap" ]; then
	    cd /afs/cern.ch/cms/tracker/sistrcalib/WWW;
	    cat ${CMSSW_BASE}/src/DQM/SiStripMonitorSummary/data/template_index_header.html | sed -e "s@insertPageName@$tag --- Tracker Maps for all IOVs ---@g" > index_new.html
	    cd $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag/plots/TrackerMap;
	    CreateIndex

	fi

    fi

    cd $WORKDIR;

done;
