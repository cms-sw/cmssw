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

# Access of all Global Tags contained in the given DB account
cmscond_tagtree_list -c frontier://cmsfrontier.cern.ch:8000/$FRONTIER/$GTACCOUNT -P /afs/cern.ch/cms/DB/conddb | grep tree | awk '{print $2}' > $GLOBALTAGCOLLECTION

TAGSUBDIR=SiStripNoise

# Loop on all Global Tags
for globaltag in `cat $GLOBALTAGCOLLECTION`; do

    echo "Processing Global Tag $globaltag";

    NEWTAG=False
    NEWIOV=False
    CFGISSAVED=False
    LOGDESTINATION=cout

    if [ `cmscond_tagtree_list -c frontier://cmsfrontier.cern.ch:8000/$FRONTIER/$GTACCOUNT -P /afs/cern.ch/cms/DB/conddb -T $globaltag | grep SiStripNoise | grep $ACCOUNT | wc -w` -eq 0 ]; then
	continue
    fi

    NOISETAG=`cmscond_tagtree_list -c frontier://cmsfrontier.cern.ch:8000/$FRONTIER/$GTACCOUNT -P /afs/cern.ch/cms/DB/conddb -T $globaltag | grep SiStripNoise | awk '{print $3}' | sed -e "s@tag:@@g"`

    if [ `echo $NOISETAG | grep Ideal | wc -w` -gt 0 ]; then
	continue
    fi

    GAINTAG=`cmscond_tagtree_list -c frontier://cmsfrontier.cern.ch:8000/$FRONTIER/$GTACCOUNT -P /afs/cern.ch/cms/DB/conddb -T $globaltag | grep SiStripGain | awk '{print $3}' | sed -e "s@tag:@@g"`

    tag=${NOISETAG}_${GAINTAG}

    # Creation of DB-Tag directory if not existing yet
    if [ ! -d "$STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR" ]; then 
	echo "Creating directory $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR"
	mkdir $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR; 
    fi

    if [ ! -d "$STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/NoiseRatios" ]; then 
	echo "Creating directory $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/NoiseRatios"
	mkdir $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/NoiseRatios; 
    fi

    if [ ! -d "$STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/NoiseRatios/$tag" ]; then 
	echo "Creating directory $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/NoiseRatios/$tag"
	mkdir $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/NoiseRatios/$tag;

	NEWTAG=True
    fi

    if [ ! -d "$STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/NoiseRatios/$tag/RelatedGlobalTags" ]; then 
	echo "Creating directory $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/NoiseRatios/$tag/RelatedGlobalTags"
	mkdir $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/NoiseRatios/$tag/RelatedGlobalTags;
    fi

    # Creation of Global Tag directory if not existing yet
    if [ ! -d "$STORAGEPATH/$DB/$ACCOUNT/$GLOBALTAGDIR" ]; then 
	echo "Creating directory $STORAGEPATH/$DB/$ACCOUNT/$GLOBALTAGDIR"
	mkdir $STORAGEPATH/$DB/$ACCOUNT/$GLOBALTAGDIR;
    fi

    if [ ! -d "$STORAGEPATH/$DB/$ACCOUNT/$GLOBALTAGDIR/$globaltag" ]; then 
	echo "Creating directory $STORAGEPATH/$DB/$ACCOUNT/$GLOBALTAGDIR/$globaltag"
	mkdir $STORAGEPATH/$DB/$ACCOUNT/$GLOBALTAGDIR/$globaltag;
    fi

    if [ ! -d "$STORAGEPATH/$DB/$ACCOUNT/$GLOBALTAGDIR/$globaltag/NoiseRatio" ]; then 
	echo "Creating directory $STORAGEPATH/$DB/$ACCOUNT/$GLOBALTAGDIR/$globaltag/NoiseRatio"
	mkdir $STORAGEPATH/$DB/$ACCOUNT/$GLOBALTAGDIR/$globaltag/NoiseRatio;
    fi

    # Creation of links between the DB-Tag and the respective Global Tags
    cd $STORAGEPATH/$DB/$ACCOUNT/$GLOBALTAGDIR/$globaltag/NoiseRatio;
    if [ -f $tag ]; then
	rm $tag;
    fi
    cat >> $tag << EOF
<html>
<body>
<a href="https://test-stripdbmonitor.web.cern.ch/test-stripdbmonitor/CondDBMonitoring/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/NoiseRatios/$tag">https://test-stripdbmonitor.web.cern.ch/test-stripdbmonitor/CondDBMonitoring/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/NoiseRatios/$tag</a>
</body>
</html>
EOF
    #ln -s $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/$tag $tag;
    cd $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/NoiseRatios/$tag/RelatedGlobalTags;
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


    # Get the list of IoVs for the given DB-Tag
    #cmscond_list_iov -c oracle://$DB/$ACCOUNT -P /afs/cern.ch/cms/DB/conddb -t $tag | awk '{if(NR>4) print "Run_In "$1 " Run_End " $2}' > list_Iov.txt  # Access via oracle
    cmscond_list_iov -c frontier://$FRONTIER/$ACCOUNT -P /afs/cern.ch/cms/DB/conddb -t $NOISETAG | awk '{if(NR>4 && $1!="Total" && $1<100000000) print "Run_In "$1 " Run_End " $2}' > list_Iov.txt # Access via Frontier

    # Access DB for the given DB-Tag and dump histograms in .png if not existing yet
    echo "Now the values are retrieved from the DB..."
    
    if [ ! -d "$STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/NoiseRatios/$tag/rootfiles" ]; then
	mkdir $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/NoiseRatios/$tag/rootfiles;
	mkdir $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/NoiseRatios/$tag/cfg;
	mkdir $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/NoiseRatios/$tag/plots;
    fi

    if [ `ls *.png | wc -w` -gt 0 ]; then
	rm *.png;
    fi

    # Process each IOV of the given DB-Tag seperately
    for IOV_number in `grep Run_In list_Iov.txt | awk '{print $2}'`; do

	if [ $IOV_number -eq 1 ]; then
	    FirstRun=$IOV_number
	    continue
	fi

	SecondRun=$IOV_number

	ROOTFILE="${tag}_Run_${IOV_number}.root"

	if [ -f $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/NoiseRatios/$tag/rootfiles/$ROOTFILE ]; then # Skip IOVs already processed. Take only new ones.
	    FirstRun=$IOV_number
	    continue
	fi

	echo "New IOV $IOV_number found. Being processed..."

	NEWIOV=True

	cat template_SiStripCorrelateNoise_cfg.py | sed -e "s@insertFirstRun@$FirstRun@g" -e "s@insertSecondRun@$SecondRun@g" -e "s@insertFrontier@$FRONTIER@g" -e "s@insertAccount@$ACCOUNT@g" -e "s@insertNoiseTag@$NOISETAG@g" -e "s@insertGainTag@$GAINTAG@g" > SiStripCorrelateNoise_cfg.py

	echo "Executing cmsRun. Stay tuned ..."

	cmsRun SiStripCorrelateNoise_cfg.py

	FirstRun=$IOV_number

	echo "cmsRun finished. Now moving the files to the corresponding directories ..."

	if [ "$NEWTAG" = "True" ] && [ "$CFGISSAVED" = "False" ]; then
	    cp SiStripCorrelateNoise_cfg.py $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/NoiseRatios/$tag/cfg/${tag}_cfg.py
	    CFGISSAVED=True
	fi

	mv correlTest.root $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/NoiseRatios/$tag/rootfiles/$ROOTFILE;

	rm out.log

	for Plot in `ls *.png`; do
	    mv $Plot $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/NoiseRatios/$tag/plots;
	done;

	#cd $WORKDIR;

    done;

    # Run the Trends and Publish all histograms on a web page
    if [ "$NEWTAG" = "True" ] || [ "$NEWIOV" = "True" ]; then

	echo "Publishing the new tag $tag (or the new IOV) on the web ..."

	cd /afs/cern.ch/cms/tracker/sistrcalib/WWW;
	cat template_index_header.html | sed -e "s@insertPageName@Noise Ratios for $NOISETAG and $GAINTAG@g" > index_new.html

	cd $STORAGEPATH/$DB/$ACCOUNT/$DBTAGDIR/$TAGSUBDIR/NoiseRatios/$tag/plots;
	CreateIndex

    fi

    cd $WORKDIR;

done;
