#!/bin/sh
# $Id: sm_compat.sh,v 1.3 2008/05/02 12:36:15 loizides Exp $

if test -e "/etc/profile.d/sm_env.sh"; then 
    source /etc/profile.d/sm_env.sh;
fi

store=/store
if test -n "$SM_STORE"; then
    store=$SM_STORE
fi

echo -n "$0 run at " >> /tmp/sm_compat.tmp
date >> /tmp/sm_compat.tmp

# notify
for i in `find $store/global/mbox/ -iname "*.work_in_progress" -cmin +10`; do
    mv $i `echo $i | cut -d. -f1`.notify
done

for i in `find $store/emulator/mbox/ -iname "*.work_in_progress" -cmin +10`; do
    mv $i `echo $i | cut -d. -f1`.notify
done

# cleanup
for i in `find $store/global/mbox/ -iname "*.smry*"`; do 
    rm -f $i; 
done

for i in `find $store/emulator/mbox/ -iname "*.smry*"`; do 
    rm -f $i; 
done

if test -e $store/global/00; then
    for i in `find $store/global/*/closed -iname "*.*_transfd"`; do 
        rm -f $i; 
    done
fi
