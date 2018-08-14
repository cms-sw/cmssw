#!/bin/bash
#for file in $(ls EcalPedestals_tree_*.db|cut -d '.' -f 1) ; do
#forls EcalPedestals_tree_*.db > list_tree_db
ls EcalPedestals_timestamp_*.db > list_tree_db
for file in `cat list_tree_db | cut -d "." -f 1` ; do
  echo $file
  cp EcalPedestals_timestamp.txt $file.txt
done
rm list_tree_db