#!/bin/sh
# $Id:$

# error file
errordir=/tmp
if test -n "$SMT0_BASE_DIR"; then
    errordir=$SMT0_BASE_DIR/error
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
    echo $0 $@ >> $errorfile
    exit 0;
fi

# send notification to T0 and check if connection was established
export T0_BASE_DIR=/nfshome0/cmsprod/TransferTest
export T0ROOT=${T0_BASE_DIR}/T0
export CONFIG=${T0_BASE_DIR}/Config/TransferSystem_Cessy.cfg
export PERL5LIB=${T0ROOT}/perl_lib:${T0_BASE_DIR}/perl

res=`${T0_BASE_DIR}/injection/sendNotification.pl --config $CONFIG $@ | grep "Connection established (3)"`
if test -z "$res"; then
    echo $0 $@ >> $errorfile
    exit 0;
fi

exit 0;
