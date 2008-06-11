#!/bin/bash
# $Id: mkstore.sh,v 1.5 2008/04/29 10:43:53 loizides Exp $

if test -e "/etc/profile.d/sm_env.sh"; then 
    source /etc/profile.d/sm_env.sh
fi

dummymode=$1
if test -n "$dummymode"; then
    dummymode=1
fi

store=/store
if test -n "$SM_STORE"; then
    store=$SM_STORE
fi

for i in emulator global; do
    cd $store 
    mkdir -p $i
    chmod 755 $i
    find $i -type l -maxdepth 1 -exec rm -f "{}" \;
    cd $store/$i;
    mkdir -p mbox && chmod 777 mbox
    mkdir -p log && chmod 777 log
    rm -rf scripts
    mkdir -p scripts && chmod 755 scripts
done

if test "$dummymode" = "1"; then
    for i in emulator global; do
        cd $store/$i/scripts
        touch dummy.pl
        chmod 755 dummy.pl
        for k in  insertFile.pl notifyTier0.pl closeFile.pl; do
            ln -fs dummy.pl $k;
        done
    done
else
    for i in emulator global; do
        cd $store/$i/scripts
        touch dummy.pl
        chmod 755 dummy.pl
        ln -s ~smpro/scripts/insertFile.pl insertFile.pl 
        ln -s ~smpro/scripts/closeFile.pl closeFile.pl 
        ln -s ~smpro/scripts/notifyTier0.pl notifyTier0.pl 
    done
fi

set counter=0
for i in `ls -d $store/sata* 2>/dev/null`; do
    cd $store
    tmount=`mount | grep $i | cut -d" " -f3`
    if test "$i" != "$tmount"; then
	echo "Warning: Omitting not mounted directory $i";
    fi
    lname=`printf "%02d" $counter`
    counter=`expr $counter + 1`
    cd $store/emulator && ln -s $i/efed $lname
    cd $store/global && ln -s $i/gcd $lname
done
