#!/bin/sh
# $Id: cleanupEmu.sh,v 1.11 2008/10/07 10:33:11 jserrano Exp $

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
LIFETIME70=60
LIFETIME65=360
LIFETIME60=720
LIFETIME50=1440
LIFETIME40=2880
LIFETIME35=4320
LIFETIME00=10080
 
# date
EPOCH=`date +%s`

#Loop over all the links 00,01,02... where SM writes
for CUD in $( ls $EMUDIR | grep ^[0-9][0-9]$ ); do
  
    # find mount point of dir to cleanup
    tmpvar=`ls -l $EMUDIR/$CUD | cut -f2 -d">"`
    if test "`hostname`" = "srv-C2D05-02" -o "`hostname`" = "srv-S2C17-01"; then
        mntpoint=`echo $tmpvar | cut -f2 -d"/"`
    else
        mntpoint=`echo $tmpvar | cut -f3 -d"/"`
    fi

    # find how full disk is to determine how much to delete
    LIFETIME=$(df | 
        awk -v LIFETIME70="$LIFETIME70" \
            -v LIFETIME65="$LIFETIME65" \
            -v LIFETIME60="$LIFETIME60" \
            -v LIFETIME50="$LIFETIME50" \
            -v LIFETIME40="$LIFETIME40" \
            -v LIFETIME35="$LIFETIME35" \
            -v LIFETIME00="$LIFETIME00" \
            -v pat="$mntpoint" \
           '$0 ~ pat {if (($5+0) > 70) print LIFETIME70; \
                 else if (($5+0) > 65) print LIFETIME65; \
                 else if (($5+0) > 60) print LIFETIME60; \
                 else if (($5+0) > 50) print LIFETIME50; \
                 else if (($5+0) > 40) print LIFETIME40; \
                 else if (($5+0) > 35) print LIFETIME35; \
                 else print LIFETIME00; }' )

    #clean
    CUDdir="$EMUDIR/$CUD/"
    find $CUDdir -cmin +$LIFETIME -type f -exec rm -f {} \; >& /dev/null

done

# clean up old notify 
find $EMUDIR/mbox/ -iname "*.notify" -cmin +15000 -type f -exec rm -f {} \; >& /dev/null
