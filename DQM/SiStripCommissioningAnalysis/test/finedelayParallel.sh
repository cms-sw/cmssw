#!/bin/sh
#paramters: input file

CASTORPATH=/castor/cern.ch/cms

echo Prestaging files...
for i in `cat $1`; { echo -n "-M $CASTORPATH$i "; } | xargs -n1000 -s129000 stager_get -U CDFDrun${1}
echo Staging `stager_qry -U CDFDrun${1} | grep STAGEIN | wc -l` files
echo `stager_qry -U CDFDrun${1} | grep STAGED | wc -l` files already staged
echo `stager_qry -U CDFDrun${1} | grep CANBEMIGR | wc -l` files still on disk
echo Starting jobs:
for fileIn in `cat $1`; do
bsub -q cmscaf1nh "`pwd`/finedelaybsub.sh $fileIn `pwd`"
done
echo Done.
echo You can monitor jobs with \"bjobs \[-w\]\". You can also use \"bpeek \[-f\] \[jobid\]\".
echo
