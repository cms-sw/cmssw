#!/bin/sh

dir=`dirname $0`

for n in $(seq 1 $1); do
    screen -d -m ${dir}/startEps.sh
done
