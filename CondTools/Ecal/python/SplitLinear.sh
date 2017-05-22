#!/bin/bash
ls Linear_*.db > list_tree_db
for file in `cat list_tree_db | cut -d "." -f 1` ; do
  echo $file
  cp Linear.txt $file.txt
done
rm list_tree_db