#!/bin/bash
# $Id:$

nfsserver=$1
filename=$2
destination=$3
parallel=$4
debug=$5;

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
    execmd="cp -a $filename $destination"
    execm2="chmod a+r $destination/`basename $filename`"
    grepstr="cp -a"
    grepstr2=$destination
fi

if test -n "$parallel"; then
    rns=`/bin/ps ax | grep "$grepstr" | grep "$grepstr2" | grep -v grep | wc -l`
    if test $rns -ge $parallel; then
        echo "Warning $0: maximum number of parallel copies ($rns), skipping for parameters $@." 
        exit 124;
    fi
fi

if test -n "`mount | grep $nfsserver`" -o -n "$debug"; then
    $execmd >/dev/null 2>&1
    if ! test $? -eq 0; then
        echo "Warning $0: error executing $execmd for parameters $@." 
    fi
    $execm2 >/dev/null 2>&1
    if ! test $? -eq 0; then
        echo "Warning $0: error executing $execm2 for parameters $@." 
    fi
else
    echo "Warning $0: error $nfsserver not mounted for parameters $@." 
fi 

exit 0
