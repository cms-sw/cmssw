#!/bin/sh
# $Id: cleanupEmu.sh,v 1.15 2011/01/31 15:45:14 gbauer Exp $

if test -e "/etc/profile.d/sm_env.sh"; then 
    source /etc/profile.d/sm_env.sh
fi

inst=`ps ax | grep "/bin/sh $0" | grep -v cron | grep -v grep | wc -l`

if test "$inst" != "2"; then
    echo "Output from ps: "
    ps ax | grep "/bin/sh $0" | grep -v cron | grep -v grep
    echo "Another instance running, exiting cleanly."
    exit 0;
fi

store=/store
if test -n "$SM_STORE"; then
    store=$SM_STORE
fi

#Path of emulator directory to cleanup
EMUDIR="$store/emulator"

if ! test -d "$EMUDIR"; then
    echo "Dir $EMUDIR not found or not a directory"
    exit 123
fi
 
# lifetime in mins / 1 day = 1440
LIFETIME45=60    # 1 hr
LIFETIME40=360   # 6 hr
LIFETIME35=720   #12 hr
LIFETIME30=1440  # 1 day
LIFETIME25=2880  # 2 day
LIFETIME20=4320  # 3 day
LIFETIME00=11520 # 8 day




 
# date
EPOCH=`date +%s`

#Loop over all the links 00,01,02... where SM writes
for CUD in $( ls $EMUDIR | grep ^[0-9][0-9]$ ); do
  
    # find mount point of dir to cleanup
    mntpoint=`ls -l $EMUDIR/$CUD | cut -f2 -d">"`


    # find how full disk is to determine how much to delete
    LIFETIME=$(df $mntpoint | 
        awk -v LIFETIME45="$LIFETIME45" \
            -v LIFETIME40="$LIFETIME40" \
            -v LIFETIME35="$LIFETIME35" \
            -v LIFETIME30="$LIFETIME30" \
            -v LIFETIME25="$LIFETIME25" \
            -v LIFETIME20="$LIFETIME20" \
            -v LIFETIME00="$LIFETIME00" \
               '/^\//{if (($5+0) > 45) print LIFETIME45; \
                 else if (($5+0) > 40) print LIFETIME40; \
                 else if (($5+0) > 35) print LIFETIME35; \
                 else if (($5+0) > 30) print LIFETIME30; \
                 else if (($5+0) > 25) print LIFETIME25; \
                 else if (($5+0) > 20) print LIFETIME20; \
                 else print LIFETIME00; }' )

    #clean
    CUDdir="$EMUDIR/$CUD/"
    find $CUDdir -cmin +$LIFETIME -type f -exec rm -f {} \; >& /dev/null

done

# clean up old notify 
find $EMUDIR/mbox/ -iname "*.notify" -cmin +15000 -type f -exec rm -f {} \; >& /dev/null
