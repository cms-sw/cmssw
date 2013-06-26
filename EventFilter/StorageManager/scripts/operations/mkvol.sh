#!/bin/bash
# $Id: mkvol.sh,v 1.9 2010/04/27 01:59:41 gbauer Exp $

## *** added security feature: 5th argument MUST BE 'makefilesys' 
## or else file system will NOT be made ***

if test -e "/etc/profile.d/sm_env.sh"; then 
    source /etc/profile.d/sm_env.sh;
fi

id=$1
sb=$2
ar=$3
vol=$4
nofs=$5

if test -z "$vol"; then
    echo "Error: $0 volname satabeast array volume"
    exit 1
fi

/sbin/multipath

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

mkdir -p $mpoint
if test "$nofs" = "makefilesys"; then
#    exstr="/sbin/mkfs.ext3 -F -T largefile -m 0 -L $mlabel $dpath"
    exstr="/sbin/mkfs.xfs -f -L $mlabel $dpath"
    echo
    echo " ***********************************************************"
    echo " ************* !! REMAKING FILE SYSTEM !! ******************"
    echo -e "Do you REALLY want to destroy files by making a new file system? (yes/no): \c "
    read  answer
    if test "$answer" = "yes"; then
	
	echo " **===>  Executing:  $exstr"
	echo " ***********************************************************"
	echo
	$exstr
	
	if test "$?" != "0"; then
	    echo "Error: Problem formatting disk, please check status!!!"
	    exit 4;
	fi	
    fi

        echo " ***********************************************************"
        echo " ***********************************************************"
fi


if ! /bin/mount -L $mlabel $mpoint; then
    echo "Error: Problem mounting disk by label, trying by mountpoint..."
    if ! /bin/mount $dpath $mpoint; then
        echo "Error: Problem mounting disk, please check status!!!"
        exit 4;
    else
        echo "Mounting directly $mpoint from $dpath succeeded."
    fi
fi

cd $mpoint
for i in efed gcd; do
    for j in open closed log; do
echo "mkdir -p $i/$j"
	mkdir -p $i/$j
	chmod 777 $i/$j;
    done
done


/sbin/multipath -F

output=$mpoint/creation_info.txt
echo "Info: `date`" >> $output
echo "Info: Volume name $id" >> $output
echo "Info: SataBeast number $sb" >> $output
echo "Info: Array number $ar" >> $output
echo "Info: Vol number $vol" >> $output
echo "Info: Found device $dpath" >> $output
echo "Info: Mount label $mlabel" >> $output
echo "Info: Mount point $mpoint" >> $output
