#!/bin/sh
# $Id:$

if test -e "/etc/profile.d/sm_env.sh"; then 
    source /etc/profile.d/sm_env.sh
fi

store=/store
if test -n "$SM_STORE"; then
    store=$SM_STORE
fi

#Path of emulator directory to cleanup
EMUDIR="$store/emulator/"
CUDBOX="$EMUDIR/mbox/"
 
# lifetime in mins / 1 day = 1440
LIFETIME90=120
LIFETIME30=360
LIFETIME20=1440
LIFETIME00=10080
 
# date
EPOCH=`date +%s`

#Loop over all the links 00,01,02... where SM writes
for CUD in $( ls $EMUDIR | grep ^[0-9][0-9]$ ); do

# find mount point of dir to cleanup
mntpoint=`ls -l $EMUDIR/$CUD | cut -f2 -d">" | cut -f3 -d"/"`

# log file
REPORT=$EMUDIR/$CUD/cleanup.txt.$EPOCH

# make sure log file does not exist
rm -f $REPORT

#Find how full disk is to determine how much to delete
LIFETIME=$(df | 
    awk -v LIFETIME90="$LIFETIME90" \
        -v LIFETIME30="$LIFETIME30" \
        -v LIFETIME20="$LIFETIME20" \
        -v LIFETIME0="$LIFETIME0" \
        -v pat="$mntpoint" \
'$0 ~ pat {if ($5 > 90) print LIFETIME90; \
           else if ($5 > 30) print LIFETIME30; \
           else if ($5 > 20) print LIFETIME20; \
           else print LIFETIME0; }' )

#clean
CUDdir="$EMUDIR/$CUD/"
find $CUDdir -cmin +$LIFETIME -not \( -type d \) -ls  -exec rm -f {} \; >& $REPORT

#Now cleanup the mbox for the files we just deleted
# log file
REPORTbox=$CUDBOX/cleanup.txt.$EPOCH
 # make sure log file does not exist
rm -rf $REPORTbox
#clean
rm -v $(awk -v cudbox="$CUDBOX" '/.dat$/ && /closed/ {fname=$0; \
                       whereclosed  = index(fname,"closed"); \
                       nopath=substr(fname,whereclosed+7,100); \
                       noext=substr(nopath,1,length(nopath)-4); \
                       print cudbox noext "smry";}' $REPORT) >> $REPORTbox 2>&1

done

