#!/bin/tcsh

echo "Shipping NAME"
cp ../DBs/NAME.db ./
/bin/sh ../dropBoxOffline.sh NAME.db NAME_T.txt 
/bin/sh ../dropBoxOffline.sh NAME.db NAME_WP.txt 
