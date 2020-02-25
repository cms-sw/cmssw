#!/bin/sh

site="T1_CH_CERN_Buffer"
lfn="/store/data/"

if [ $# -ne 2 ]
then
  echo "usage: $0 <site> [<LFN>]"
fi

if [ $# -eq 0 ] ; then exit 1; fi

if [ $# -eq 1 ]
then
  site=$1
  lfn="/store/data"
else
  site=$1
  lfn=$2
fi

curl -ks "https://cmsweb.cern.ch/phedex/datasvc/perl/prod/lfn2pfn?node=${site}&lfn=${lfn}&protocol=srmv2" | grep PFN | cut -d "'" -f4


