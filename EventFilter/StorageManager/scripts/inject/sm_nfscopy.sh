#!/bin/bash
# $Id: sm_nfscopy.sh,v 1.10 2012/03/16 19:11:28 babar Exp $

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
    execmd2="echo done sleeping"
    grepstr=$execmd
    grepstr2=$grepstr
else
    fname=`basename $filename`
    execmd="cp -a $filename $destination/${fname}.part"
    execmd2="mv $destination/${fname}.part $destination/$fname"
    execmd3="chmod a+r $destination/$fname"
    grepstr="cp -a"
    grepstr2=$destination
fi

if test -n "$parallel"; then
    # XXX Replace this by a decent pgrep on PPID
    rns=`/bin/ps ax | grep "$grepstr" | grep "$grepstr2" | grep -v grep | wc -l`
    if test $rns -ge $parallel; then
        echo "Warning $0: maximum number of parallel copies ($rns), skipping for parameters $@." 
        exit 124;
    fi
fi

if test -n "$debug" -o "$nfsserver" = "local" -o -n "`mount | grep $nfsserver`"; then
    if ! $execmd; then
        echo "Warning $0: error executing $execmd for parameters $@." >&2
        exit 125;
    fi
    if ! $execmd2; then
        echo "Warning $0: error executing $execmd2 for parameters $@." >&2
        exit 126;
    fi
    $execmd3
else
    echo "Warning $0: error $nfsserver not mounted for parameters $@." >&2
    exit 127;
fi 

exit 0
