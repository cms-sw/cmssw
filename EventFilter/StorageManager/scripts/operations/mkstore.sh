#!/bin/bash
# $Id: mkstore.sh,v 1.10 2012/07/06 16:00:16 gbauer Exp $

if test -e "/etc/profile.d/sm_env.sh"; then 
    source /etc/profile.d/sm_env.sh
fi

store=/store
if test -n "$SM_STORE"; then
    store=$SM_STORE
fi



for i in emulator global; do
    cd $store 
    if [ ! -d $i ]; then
        mkdir -p $i
        chmod 755 $i
#        find $i -type l -maxdepth 1 -exec rm -f "{}" \;
        find $i  -maxdepth 1 -type l -exec rm -f "{}" \;
        cd $store/$i;
#        mkdir -p mbox && chmod 777 mbox
#         ln -s ./00/log/ log
#        mkdir -p log && chmod 777 log
        rm -rf scripts
        mkdir -p scripts && chmod 755 scripts

        # Dealing with scripts
        cd $store/$i/scripts
        touch dummy.pl
        chmod 755 dummy.pl
        for k in  insertFile.pl notifyTier0.pl closeFile.pl; do
            ln -fs dummy.pl $k;
        done
    fi
done


set -- `ls -d $store/sata* 2>/dev/null`
if [ $# = 1 ]; then
    echo "Faking multiple disks"
    set -- $1 $1 $1 $1
fi

let counter=0
for i in "$@"; do
    cd $store
    tmount=`mount | grep $i | cut -d" " -f3`
    if test "$i" != "$tmount"; then
	echo "Warning: Omitting not mounted directory $i";
        continue;
    fi
    lname=`printf "%02d" $counter`
    counter=`expr $counter + 1`
    cd $store/emulator && ln -s $i/efed $lname 
    cd $store/global   && ln -s $i/gcd  $lname 
done
    cd $store/emulator && ln -s ./00/log/ log
    cd $store/global   && ln -s ./00/log/ log
