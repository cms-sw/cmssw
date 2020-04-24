#!/bin/bash

date
# needed to allow the loop on *.png without using "*.png" as value
shopt -s nullglob

if [ $# -ne 2 ]; then
    afstokenchecker.sh "You have to provide a <tag-search-string>, a <DB>!!!"
    exit
fi

afstokenchecker.sh "Starting execution of Monitor_GlobalTags_V2 $1 $2"

#Example: SEARCHSTRING=SiStrip
SEARCHSTRING=$1
#Example: DB=cms_orcoff_prod
DB=$2
DBTAGCOLLECTION=DBTagCollection.txt
GLOBALTAGCOLLECTION=GlobalTags.txt
DBTAGDIR=DBTagCollection
GLOBALTAGDIR=GlobalTags
STORAGEDIR=CondDBMonitoring
STORAGEPATH=/afs/cern.ch/cms/tracker/sistrcalib/WWW/$STORAGEDIR
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
conddb --db $DB --nocolors listGTs | grep " GT " | awk '{print $1}' > $GLOBALTAGCOLLECTION


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
    conddb --db $DB --nocolors list $globaltag | grep $SEARCHSTRING > $DBTAGCOLLECTION

    while read tagstring; do

	if [ `echo $tagstring | wc -w` -eq 0 ]; then
	    continue
	fi

	TAG=`echo $tagstring | awk '{print $3}'`
	RECORD=`echo $tagstring | awk '{print $1}'`
	LABEL=`echo $tagstring | awk '{print $2}'`
	OBJECT="ToBeDefined"
#	OBJECT=`echo $tagstring | awk '{print $3}' | sed -e "s@object:@@g"`
	ACCOUNT="CMS_CONDITIONS"
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
	    
	    RECORDANDOBJECTNAME="Record name: $RECORD Label name: $LABEL Object Name: $OBJECT"
	    echo $RECORDANDOBJECTNAME 
	    
	    rm -f $TAG; 
	    cat >> $TAG << EOF
<html>
<body>
<a href="https://test-stripdbmonitor.web.cern.ch/test-stripdbmonitor/$STORAGEDIR/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$TAG">https://test-stripdbmonitor.web.cern.ch/test-stripdbmonitor/$STORAGEDIR/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$TAG</a>
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
<a href="https://test-stripdbmonitor.web.cern.ch/test-stripdbmonitor/$STORAGEDIR/$DB/$GLOBALTAGDIR/$globaltag">https://test-stripdbmonitor.web.cern.ch/test-stripdbmonitor/$STORAGEDIR/$DB/$GLOBALTAGDIR/$globaltag</a>
</body>
</html>
EOF
	fi
	
	cd $WORKDIR;
	
    done < $DBTAGCOLLECTION;
    
    
done;
