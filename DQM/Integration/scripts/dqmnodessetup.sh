#!/bin/bash

machines="srv-c2d05-10 \
srv-c2d05-11 srv-c2d05-12 srv-c2d05-13 srv-c2d05-14 \
srv-c2d05-15 srv-c2d05-16 srv-c2d05-17 srv-c2d05-18 \
srv-c2d05-19"

user=dqmdev

for c in $machines; do ssh $c -t mkdir /home/${USER}local/output ; done
for c in $machines; do ssh $c -t mkdir /home/${USER}local/reference ; done
for c in $machines; do ssh $c -t chmod a+rwx /home/${USER}local ; done
for c in $machines; do ssh $c -t chmod a+rwx /home/${USER}local/output ; done
for c in $machines; do ssh $c -t chmod a+rwx /home/${USER}local/reference ; done


