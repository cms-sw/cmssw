#!/bin/sh


#script to clean up LOOKAREA --- to run on cms-data-lookarea
#(derived from  cleanupEmu.sh)


LOCALNODE=`hostname | tr '[A-Z]' '[a-z]'` 

ISLOOKNODE=`host cms-data-lookarea | grep -ic $LOCALNODE`
if  test "$ISLOOKNODE" -eq  "0"; then
    echo "  "
    echo "        **** WRONG NODE: $LOCALNODE is not cms-data-lookarea ***  "
    echo "  "
    exit 12
fi


#check if any previous invocation of this script is still ongoing:    
inst=`ps ax | grep "/bin/sh $0" | grep -v cron | grep -v grep | wc -l`    
if test "$inst" != "2"; then
    echo "Output from ps: "
    ps ax | grep "/bin/sh $0" | grep -v cron | grep -v grep
    echo "Another instance running, exiting cleanly."
    exit 0;
fi


#Path of LOOKAREA---directory to cleanup:
LOOKDIR="/lookarea_SM"
if ! test -d "$LOOKDIR"; then
    echo "Dir $LOOKDIR not found or not a directory"
    exit 123
fi



#Delete BIG/SMALL files with different ages,
# cut-off size between BIG and SMALL files, in kilobytes:
SMALLFILESIZE=35840

#convert to MB for human convenience:
SMALLFILESIZE_M=$(echo "scale=2;$SMALLFILESIZE/1024 " | bc) 



#deletion cutoffs:
# lifetime in mins / 1 day = 1440
LIFETIME_BIG85=60      #  1 hr
LIFETIME_BIG80=180     #  3 hr
LIFETIME_BIG75=360     #  6 hrs
LIFETIME_BIG70=720     # 12 hrs 
LIFETIME_BIG60=1440    # 1 day
LIFETIME_BIG50=2880    # 2 days
LIFETIME_BIG40=4320    # 3 days
LIFETIME_BIG30=7200    # 5 days 
LIFETIME_BIG20=10080   # 7 days
LIFETIME_BIG10=20160   #14 days
LIFETIME_BIG00=50400   #35 days


LIFETIME_SMLL90=120     #  2 hrs
LIFETIME_SMLL85=360     #  6 hrs
LIFETIME_SMLL80=720     # 12 hrs 
LIFETIME_SMLL70=1440    # 1 day
LIFETIME_SMLL60=4320    # 3 days
LIFETIME_SMLL50=7200    # 5 days 
LIFETIME_SMLL30=10800   # 7 days
LIFETIME_SMLL18=20160   #14 days
LIFETIME_SMLL00=50400   #35 days





mntpoint="$LOOKDIR"

# delete big files first:

    # find how full disk is to determine how much to delete
LIFETIME_BIG=$(df | 
    awk -v LIFETIME_BIG85="$LIFETIME_BIG85" \
	-v LIFETIME_BIG80="$LIFETIME_BIG80" \
	-v LIFETIME_BIG75="$LIFETIME_BIG75" \
	-v LIFETIME_BIG70="$LIFETIME_BIG70" \
	-v LIFETIME_BIG60="$LIFETIME_BIG60" \
	-v LIFETIME_BIG50="$LIFETIME_BIG50" \
	-v LIFETIME_BIG40="$LIFETIME_BIG40" \
	-v LIFETIME_BIG30="$LIFETIME_BIG30" \
	-v LIFETIME_BIG20="$LIFETIME_BIG20" \
	-v LIFETIME_BIG10="$LIFETIME_BIG10" \
	-v LIFETIME_BIG00="$LIFETIME_BIG00" \
	-v pat="$mntpoint"  \
	'$0 ~ pat {if ($4 > 85) print LIFETIME_BIG85; \
	else if ($4 > 80) print LIFETIME_BIG80; \
	else if ($4 > 75) print LIFETIME_BIG75; \
	else if ($4 > 70) print LIFETIME_BIG70; \
	else if ($4 > 60) print LIFETIME_BIG60; \
	else if ($4 > 50) print LIFETIME_BIG50; \
	else if ($4 > 40) print LIFETIME_BIG40; \
	else if ($4 > 30) print LIFETIME_BIG30; \
	else if ($4 > 20) print LIFETIME_BIG20; \
	else if ($4 > 10) print LIFETIME_BIG10; \
	else print LIFETIME_BIG00; }' )


    #clean BIG
CUDdir="$LOOKDIR"
NDELETEDBIG=`find $CUDdir -cmin +$LIFETIME_BIG -type f -a -size +"$SMALLFILESIZE"k  -a  -exec rm -fv {} \; | grep -c removed`



DELETETIME_hr=$(echo "scale=2;$LIFETIME_BIG/60 " | bc) 
DELETETIME_day=$(echo "scale=1;$DELETETIME_hr/24 " | bc) 

echo " >>>> Deleted $NDELETEDBIG files older than $LIFETIME_BIG ($DELETETIME_hr hrs; or $DELETETIME_day days) and larger than $SMALLFILESIZE kB ($SMALLFILESIZE_M MB)"


# then delete small files:

    # find how full disk is to determine how much to delete
LIFETIME_SMLL=$(df | 
    awk -v LIFETIME_SMLL90="$LIFETIME_SMLL90" \
	-v LIFETIME_SMLL85="$LIFETIME_SMLL85" \
	-v LIFETIME_SMLL80="$LIFETIME_SMLL80" \
	-v LIFETIME_SMLL70="$LIFETIME_SMLL70" \
	-v LIFETIME_SMLL60="$LIFETIME_SMLL60" \
	-v LIFETIME_SMLL50="$LIFETIME_SMLL50" \
	-v LIFETIME_SMLL30="$LIFETIME_SMLL30" \
	-v LIFETIME_SMLL18="$LIFETIME_SMLL18" \
	-v LIFETIME_SMLL00="$LIFETIME_SMLL00" \
	-v pat="$mntpoint" \
	'$0 ~ pat {if ($4 > 90) print LIFETIME_SMLL90; \
	else if ($4 > 85) print LIFETIME_SMLL85; \
	else if ($4 > 80) print LIFETIME_SMLL80; \
	else if ($4 > 70) print LIFETIME_SMLL70; \
	else if ($4 > 60) print LIFETIME_SMLL60; \
	else if ($4 > 50) print LIFETIME_SMLL50; \
	else if ($4 > 30) print LIFETIME_SMLL30; \
	else if ($4 > 18) print LIFETIME_SMLL18; \
	else print LIFETIME_SMLL00; }' )


    #clean SMALL files
#    CUDdir="$LOOKDIR"

#increment threshold cuz tests are only > or <, not =
SMALLFILESIZE=$(echo "scale=0;$SMALLFILESIZE+1 " | bc) 


NDELETEDSMLL=`find $CUDdir -cmin +$LIFETIME_SMLL -type f -a -size -"$SMALLFILESIZE"k  -a  -exec rm -fv {}  \; | grep -c removed`



DELETETIME_hr=$(echo "scale=2;$LIFETIME_SMLL/60 " | bc) 
DELETETIME_day=$(echo "scale=1;$DELETETIME_hr/24 " | bc) 
echo " >>>> Deleted $NDELETEDSMLL files older than $LIFETIME_SMLL ($DELETETIME_hr hrs; or $DELETETIME_day days) and smaller than  $SMALLFILESIZE kB ($SMALLFILESIZE_M MB)"


exit 0;


