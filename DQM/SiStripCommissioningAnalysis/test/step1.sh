#!/bin/sh
#paramters: runnumber partition ishalow ishahigh ishastep vfslow vfshigh vfsstep
for i in `nsls /castor/cern.ch/user/d/delaer/CMStracker/${1} | grep SiStripCommissioningSource`; { echo -n "-M /castor/cern.ch/user/d/delaer/CMStracker/${1}/$i "; } | xargs -n1000 -s129000 stager_get -U CDrun${1}
echo Staging `stager_qry -U CDrun${1} | grep STAGEIN | wc -l` files
echo `stager_qry -U CDrun${1} | grep STAGED | wc -l` files already staged
echo `stager_qry -U CDrun${1} | grep CANBEMIGR | wc -l` files still on disk
for isha in `seq $3 $5 $4`; do
for vfs  in `seq $6 $8 $7`; do
bsub -q cmscaf "`pwd`/step1bsub.sh ${1} ${2} ${isha} ${vfs} `pwd`"
done
done

