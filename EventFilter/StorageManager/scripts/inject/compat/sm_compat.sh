#!/bin/sh
# $Id:$

if test -e "/etc/profile.d/sm_env.sh"; then 
    source /etc/profile.d/sm_env.sh;
fi

store=/store
if test -n "$SM_STORE"; then
    store=$SM_STORE
fi

date >> /tmp/sm_compat.tmp

for i in `find $store/global/mbox/ -iname "*.work_in_progress" -cmin +10`; do
    mv $i `echo $i | cut -d. -f1`.notify
done

for i in `find $store/emulator/mbox/ -iname "*.work_in_progress" -cmin +10`; do
    mv $i `echo $i | cut -d. -f1`.notify
done

for i in `find $store/global/mbox/ -iname "*.smry*"`; do 
    rm -f $i; 
done

for i in `find $store/emulator/mbox/ -iname "*.smry*"`; do 
    rm -f $i; 
done
