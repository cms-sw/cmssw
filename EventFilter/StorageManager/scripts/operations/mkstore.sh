#!/bin/bash
#$Id:$

dummymode=$1
if test -n "$dummymode"; then
    dummymode=1;
fi

for i in emulator global; do
    cd /store 
    mkdir -p $i
    chmod 777 $i
    find $i -type l -maxdepth 1 -exec rm -f "{}" \;
    cd /store/$i;
    mkdir -p mbox && chmod 777 mbox
    rm -rf scripts
    mkdir -p scripts && chmod 755 scripts
done

if test "$dummymode" = "1"; then
    for i in emulator global; do
        cd scripts
        touch dummy.pl
        chmod 755 dummy.pl
        for k in  insertFile.pl notifyTier0.pl closeFile.pl; do
            ln -fs dummy.pl $k;
        done
    done
else
    cd /store/emulator/scripts
    ln -s ~smpro/scripts/emulatorOnly.pl dummy.pl
    for k in  insertFile.pl notifyTier0.pl closeFile.pl; do
        ln -fs dummy.pl $k;
    done
    cd /store/global/scripts
    touch dummy.pl
    chmod 755 dummy.pl
    ln -s ~smpro/scripts/insertFile.pl insertFile.pl 
    ln -s ~smpro/scripts/closeFile.pl closeFile.pl 
    ln -s ~smpro/scripts/notifyTier0.pl notifyTier0.pl 
fi

set counter=0
for i in `ls -d /store/sata* 2>/dev/null`; do
    cd /store
    tmount=`mount | grep $i | cut -d" " -f3`
    if test "$i" != "$tmount"; then
	echo "Warning: Omitting not mounted directory $i";
    fi
    lname=`printf "%02d" $counter`
    counter=`expr $counter + 1`
    cd /store/emulator && ln -s $i/efed $lname
    cd /store/global && ln -s $i/gcd $lname
done
