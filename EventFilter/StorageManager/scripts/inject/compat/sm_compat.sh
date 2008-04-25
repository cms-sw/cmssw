#!/bin/sh

date >> /tmp/sm_compat.tmp

for i in `find /store/global/mbox/ -iname "*.work_in_progress" -cmin +10`; do
    mv $i `echo $i | cut -d. -f1`.notify
done

for i in `find /store/emulator/mbox/ -iname "*.work_in_progress" -cmin +10`; do
    mv $i `echo $i | cut -d. -f1`.notify
done

for i in `find /store/global/mbox/ -iname "*.smry*"`; do 
    rm -f $i; 
done

for i in `find /store/emulator/mbox/ -iname "*.smry*"`; do 
    rm -f $i; 
done
