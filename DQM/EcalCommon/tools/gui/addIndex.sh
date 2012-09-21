#!/bin/bash

if [ ${HOSTNAME} != "srv-S2F19-29" ]; then
    echo "This is not ecalod-disk01 !!!"
    exit 1
fi

guidir=/data/ecalod-disk01/dqm-gui
datadir=/data/ecalod-disk01/dqm-data

ix=$guidir/state/dqmgui/online/ix
tmp=$datadir/tmp/closed
dest=$datadir/root

cd $guidir

. $PWD/current/apps/dqmgui/etc/profile.d/env.sh

for file in $(ls --color=never $tmp/DQM* 2> /dev/null); do
    echo "Adding index for "$(echo $file | sed 's|^.*/\([^\/]*\)$|\1|')
    visDQMIndex add $ix $file && mv $file $dest/
done
