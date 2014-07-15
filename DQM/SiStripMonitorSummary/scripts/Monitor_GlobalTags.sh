#!/bin/bash

date
# needed to allow the loop on *.png without using "*.png" as value
shopt -s nullglob

if [ $# -ne 4 ]; then
    afstokenchecker.sh "You have to provide a <tag-search-string>, a <DB>, a <GTAccount> and the <FrontierPath> !!!"
    exit
fi

afstokenchecker.sh "Starting execution of Monitor_GlobalTags $1 $2 $3 $4"

#Example: SEARCHSTRING=SiStrip
SEARCHSTRING=$1
#Example: DB=cms_orcoff_prod
DB=$2
#Example: GTACCOUNT=CMS_COND_21X_GLOBALTAG
GTACCOUNT=$3
#Example: FRONTIER=FrontierProd
FRONTIER=$4
DBTAGCOLLECTION=DBTagCollection.txt
GLOBALTAGCOLLECTION=GlobalTags.txt
DBTAGDIR=DBTagCollection
GLOBALTAGDIR=GlobalTags
STORAGEPATH=/afs/cern.ch/cms/tracker/sistrcalib/WWW/CondDBMonitoring
WORKDIR=$PWD

# Creation of all needed directories if not existing yet
if [ ! -d "$STORAGEPATH/$DB" ]; then 
    afstokenchecker.sh "Creating directory $STORAGEPATH/$DB"
    mkdir $STORAGEPATH/$DB;
fi

if [ ! -d "$STORAGEPATH/$DB/$GLOBALTAGDIR" ]; then 
    afstokenchecker.sh "Creating directory $STORAGEPATH/$DB/$GLOBALTAGDIR"
    mkdir $STORAGEPATH/$DB/$GLOBALTAGDIR; 
fi

# Access of all Global Tags contained in the given DB account
afstokenchecker.sh "Preparing list of Global Tags"
rm -rf $GLOBALTAGCOLLECTION
cmscond_tagtree_list -c frontier://cmsfrontier.cern.ch:8000/$FRONTIER/$GTACCOUNT -P /afs/cern.ch/cms/DB/conddb | grep tree | awk '{print $2}' > $GLOBALTAGCOLLECTION


# Loop on all Global Tags
for globaltag in `cat $GLOBALTAGCOLLECTION`; do

    afstokenchecker.sh "Processing Global Tag $globaltag";

    if [ -d "$STORAGEPATH/$DB/$GLOBALTAGDIR/$globaltag" ]; then
# already known GT: check if the directory is empty otherwise skip it
	NOTONLYNOISERATIOS=false
	for file in `ls $STORAGEPATH/$DB/$GLOBALTAGDIR/$globaltag/`; do
	    if [ $file != "NoiseRatios" ] && [ $file != "RunInfo" ]; then
		NOTONLYNOISERATIOS=true
		continue
	    fi
	done
	if [ "$NOTONLYNOISERATIOS" == "true" ]; then
	    continue
	fi
	afstokenchecker.sh "Directory $STORAGEPATH/$DB/$GLOBALTAGDIR/$globaltag contains only NoiseRatios: to be processed";
    else
	afstokenchecker.sh "Creating directory $STORAGEPATH/$DB/$GLOBALTAGDIR/$globaltag";
	mkdir $STORAGEPATH/$DB/$GLOBALTAGDIR/$globaltag;
    fi
	
    rm -f $DBTAGCOLLECTION
    cmscond_tagtree_list -c frontier://cmsfrontier.cern.ch:8000/$FRONTIER/$GTACCOUNT -P /afs/cern.ch/cms/DB/conddb -T $globaltag | grep $SEARCHSTRING | awk '{printf "%s %s %s %s\n",$3,$4,$5,$7}' > $DBTAGCOLLECTION

    while read tagstring; do

	if [ `echo $tagstring | wc -w` -eq 0 ]; then
	    continue
	fi

	TAG=`echo $tagstring | awk '{print $1}' | sed -e "s@tag:@@g"`
	RECORD=`echo $tagstring | awk '{print $2}' | sed -e "s@record:@@g"`
	OBJECT=`echo $tagstring | awk '{print $3}' | sed -e "s@object:@@g"`
#	ACCOUNT=`echo $tagstring | awk '{print $4}' | sed -e "s@pfn:frontier://$FRONTIER/@@g"`
	ACCOUNT=`echo $tagstring | awk '{print $4}' | awk 'BEGIN {FS = "/" } ; { print $NF }'`
	TAGSUBDIR="Unknown"

    if      [ `echo $TAG | grep "Noise" | wc -w` -gt 0 ]; then
	TAGSUBDIR=SiStripNoise
    else if [ `echo $TAG | grep "Pedestal" | wc -w` -gt 0 ]; then
	TAGSUBDIR=SiStripPedestal
    else if [ `echo $TAG | grep "Gain" | wc -w` -gt 0 ]; then
	TAGSUBDIR=SiStripApvGain
    else if [ `echo $TAG | grep "Bad" | wc -w` -gt 0 ]; then
	TAGSUBDIR=SiStripBadChannel
    else if [ `echo $TAG | grep "Cabling" | wc -w` -gt 0 ]; then
	TAGSUBDIR=SiStripFedCabling
    else if [ `echo $TAG | grep "Lorentz" | wc -w` -gt 0 ]; then
	TAGSUBDIR=SiStripLorentzAngle
    else if [ `echo $TAG | grep "BackPlane" | wc -w` -gt 0 ]; then
	TAGSUBDIR=SiStripBackPlaneCorrection
    else if [ `echo $TAG | grep "Threshold" | wc -w` -gt 0 ]; then
	TAGSUBDIR=SiStripThreshold
    else if [ `echo $TAG | grep "VOff" | wc -w` -gt 0 ]; then
	TAGSUBDIR=SiStripVoltage
    else if [ `echo $TAG | grep "Latency" | wc -w` -gt 0 ]; then
	TAGSUBDIR=SiStripLatency
    else if [ `echo $TAG | grep "Shift" | wc -w` -gt 0 ]; then
	TAGSUBDIR=SiStripShiftAndCrosstalk
    else if [ `echo $TAG | grep "APVPhaseOffsets" | wc -w` -gt 0 ]; then
	TAGSUBDIR=SiStripAPVPhaseOffsets
    else if [ `echo $TAG | grep "AlCaRecoTriggerBits" | wc -w` -gt 0 ]; then
	TAGSUBDIR=SiStripDQM
    else
	TAGSUBDIR=Unknown
	afstokenchecker.sh "Unknown tag type $TAG: skipped";
	continue
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
#	echo "$tagstring";
#	echo "$TAG $RECORD $OBJECT $ACCOUNT $TAGSUBDIR";

	# Creation of links between the DB-Tag and the respective Global Tags
	if [ ! -d "$STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$TAG" ]; then
	    afstokenchecker.sh "Tag $TAG is unknown: skipped";
	    echo "$tagstring";
	    continue
	fi
	if [ ! -d "$STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$TAG/RelatedGlobalTags" ]; then
	    afstokenchecker.sh "Creating directory $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$TAG/RelatedGlobalTags";
	    mkdir $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$TAG/RelatedGlobalTags;
	fi
	if [ ! -f $STORAGEPATH/$DB/$GLOBALTAGDIR/$globaltag/$TAG ] || [ ! -f $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$TAG/RelatedGlobalTags/$globaltag ]; then
	    afstokenchecker.sh "Creating a link between $globaltag and $TAG";

	    cd $STORAGEPATH/$DB/$GLOBALTAGDIR/$globaltag;
	    
	    RECORDANDOBJECTNAME="Record name: $RECORD Object Name: $OBJECT"
	    echo $RECORDANDOBJECTNAME 
	    
	    rm -f $TAG; 
	    cat >> $TAG << EOF
<html>
<body>
<a href="https://test-stripdbmonitor.web.cern.ch/test-stripdbmonitor/CondDBMonitoring/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$TAG">https://test-stripdbmonitor.web.cern.ch/test-stripdbmonitor/CondDBMonitoring/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$TAG</a>
<br />
$RECORDANDOBJECTNAME
</body>
</html>
EOF
	    
	    cd $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$TAG/RelatedGlobalTags;
	    
	    rm -f $globaltag;
	    cat >> $globaltag << EOF
<html>
<body>
<a href="https://test-stripdbmonitor.web.cern.ch/test-stripdbmonitor/CondDBMonitoring/$DB/$GLOBALTAGDIR/$globaltag">https://test-stripdbmonitor.web.cern.ch/test-stripdbmonitor/CondDBMonitoring/$DB/$GLOBALTAGDIR/$globaltag</a>
</body>
</html>
EOF
	fi
	
	cd $WORKDIR;
	
    done < $DBTAGCOLLECTION;
    
    
done;
