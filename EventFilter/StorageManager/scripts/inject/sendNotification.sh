#!/bin/sh
# $Id: sendNotification.sh,v 1.7 2008/11/06 00:42:49 loizides Exp $

# error dir and file
errordir=/tmp
if test -n "$SMT0_LOCAL_RUN_DIR"; then
    errordir=$SMT0_LOCAL_RUN_DIR/logs
fi
mkdir -p $errordir

hname=`hostname | cut -d. -f1`
dname=`date "+%Y%m%d"`
if test -n "$SMIW_RUNNUM"; then
    errorfile=$errordir/$dname-$hname-iw-${SMIW_RUNNUM}.sendnotify
else
    errorfile=$errordir/$dname-$hname-iw-$$.sendnotify
fi

# test if copyworker is running
cpw=`ps ax | grep -i CopyWorker | grep -v grep`
if test -z "$cpw"; then
    echo "#Error: Copyworker is not running" >> $errorfile
    echo $0 $@ >> $errorfile
    exit 0;
fi

# send notification to T0 and check if connection was established
if test -z "$T0_BASE_DIR"; then
    export T0_BASE_DIR=/nfshome0/cmsprod/TransferTest
    export T0ROOT=${T0_BASE_DIR}/T0
    export T0_CONFIG=${T0_BASE_DIR}/Config/TransferSystem_Cessy.cfg
    export PERL5LIB=${T0ROOT}/perl_lib:${T0_BASE_DIR}/perl
    export T0_NOTIFY=${T0ROOT}/operations/sendNotification.pl
fi

res=`${T0_NOTIFY} --config $T0_CONFIG $@ 2>&1 | grep "Connection established (3)"`
if test -z "$res"; then
    echo "#Error: Most likely CopyManager is not running (cmsusr3)" >> $errorfile
    echo $0 $@ >> $errorfile
    exit 0;
fi

exit 0;
