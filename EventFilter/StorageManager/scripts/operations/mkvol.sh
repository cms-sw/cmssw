#!/bin/bash
# $Id: mkvol.sh,v 1.4 2008/06/11 13:58:04 loizides Exp $

if test -e "/etc/profile.d/sm_env.sh"; then 
    source /etc/profile.d/sm_env.sh;
fi

id=$1
sb=$2
ar=$3
vol=$4

if test -z "$vol"; then
    echo "Error: $0 volname satabeast array volume"
    exit 1
fi

dpath=/dev/mapper/`/sbin/multipath -l | grep -i $id | cut -d" " -f1`
mlabel=sata`printf "%02d" $sb`a`printf "%02d" $ar`v`printf "%02d" $vol`
mpoint=/store/$mlabel
if test -n "$SM_STORE"; then
    mpoint=$SM_STORE/$mlabel
fi

echo "Info: Volume name $id"
echo "Info: SataBeast number $sb"
echo "Info: Array number $ar"
echo "Info: Vol number $vol"
echo "Info: Found device $dpath"
echo "Info: Mount point $mpoint"
echo "Info: LABEL=$id $mpoint xfs defaults 1 2"

if ! test -b $dpath; then
    echo "Error: $dpath not found to be block device"
    exit 2;
fi

testdpath=`mount | grep $dpath | cut -d" " -f1`
if test "$dpath" = "$testdpath"; then
    echo "Error: $dpath is mounted, please unmount first!!!"
    exit 3;
fi

testmpoint=`mount | grep $mpoint | cut -d" " -f1`
if test "$mpoint" = "$testmpoint"; then
    echo "Error: $mpoint is used, please unmount first!!!"
    exit 3;
fi

exstr="mkdir -p $mpoint && /sbin/mkfs.ext3 -F -T largefile -m 0 -L $mlabel $dpath && /bin/mount -L $mlabel $mpoint"
echo "Executing $exstr"
echo
 
mkdir -p $mpoint && /sbin/mkfs.ext3 -F -T largefile -m 0 -L $mlabel $dpath && /bin/mount -L $mlabel $mpoint

if test "$?" != "0"; then
    echo "Error: Problem formatting disk, please check status!!!"
    exit 4;
fi

cd $mpoint
for i in efed gcd; do
    for j in open closed; do
echo "mkdir -p $i/$j"
	mkdir -p $i/$j
	chmod 777 $i/$j;
    done
done

output=$mpoint/creation_info.txt
echo "Info: `date`" >> $output
echo "Info: Volume name $id" >> $output
echo "Info: SataBeast number $sb" >> $output
echo "Info: Array number $ar" >> $output
echo "Info: Vol number $vol" >> $output
echo "Info: Found device $dpath" >> $output
echo "Info: Mount label $mlabel" >> $output
echo "Info: Mount point $mpoint" >> $output
