#!/bin/sh
DATADIR=.

DSTART=`python -c 'from datetime import *; today = date.today(); print today - timedelta(days=2)'`
DSTOP=`python -c 'from datetime import *; today = date.today(); print today + timedelta(days=1)'`
NAME=$DATADIR/dashboard_status_`date +%y-%m-%d_%T`.xml

# mv $DATADIR/dashboard_status.xml $NAME
# gzip $NAME

curl -H 'Accept: text/xml' 'http://dashb-cms-job.cern.ch/dashboard/request.py/jobsummary-plot-or-table?user=&site=&ce=&submissiontool=&dataset&application=&rb=&activity=&grid=&date1='$DSTART'&date2='$DSTOP'&sortby=site&nbars=&jobtype=&tier=&status=pending&status=running&status=donesuccess&check=submitted' > $DATADIR/dashboard_status.xml

