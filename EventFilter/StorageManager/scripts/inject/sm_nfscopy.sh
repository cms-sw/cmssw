#!/bin/bash
# $Id: sm_nfscopy.sh,v 1.5 2008/11/04 16:18:14 loizides Exp $

nfsserver=$1
filename=$2
destination=$3
parallel=$4
debug=$5;

# main starts here
if test -z "$3"; then
    echo "Usage: $0 nfsserver filename destination [parallel] [debug]"
    exit 123;
fi

if test -n "$debug"; then
    execmd="sleep 15"
    execm2="echo done sleeping"
    grepstr=$execmd
    grepstr2=$grepstr
else
    fname=`basename $filename`
    execmd="cp -a $filename $destination/${fname}.part"
    execm2="mv $destination/${fname}.part $destination/$fname"
    execm3="chmod a+r $destination/$fname"
    grepstr="cp -a"
    grepstr2=$destination
fi

if test -z "$3"; then
    echo "Usage: $0 nfsserver filename destination [parallel] [debug]"
    exit 123;
fi

if test -n "$parallel"; then
    rns=`/bin/ps ax | grep "$grepstr" | grep "$grepstr2" | grep -v grep | wc -l`
    if test $rns -ge $parallel; then
        echo "Warning $0: maximum number of parallel copies ($rns), skipping for parameters $@." 
        exit 124;
    fi
fi

if test -n "$debug" -o "$nfsserver" = "local" -o -n "`mount | grep $nfsserver`"; then
    $execmd >/dev/null 2>&1
    if ! test $? -eq 0; then
        echo "Warning $0: error executing $execmd for parameters $@." 
        exit 125;
    fi
    $execm2 >/dev/null 2>&1
    if ! test $? -eq 0; then
        echo "Warning $0: error executing $execm2 for parameters $@." 
        exit 126;
    fi
    $execm3 >/dev/null 2>&1
else
    echo "Warning $0: error $nfsserver not mounted for parameters $@." 
    exit 127;
fi 

exit 0
