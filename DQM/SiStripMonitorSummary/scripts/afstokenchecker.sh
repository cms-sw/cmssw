#!/bin/bash

tokenstring=`tokens`
mindelta=7200

#echo "tokenstring $tokenstring"

start=`expr index "$tokenstring" [`
stop=`expr index "$tokenstring" ]`

#echo "start $start $stop"

exptime=`date --date="${tokenstring:$start+8:$stop-$start-9}" +"%s"`
nowtime=`date +"%s"`
delta=`expr $exptime - $nowtime`
#echo "$delta"

if [ $delta -le $mindelta ]; then
    date "+[%c] $1 . Token expires in $delta second: to be renewed"
/usr/sue/bin/kinit -R
    tokens
else
    date "+[%c] $1 . Token expires in $delta second: ok"
fi