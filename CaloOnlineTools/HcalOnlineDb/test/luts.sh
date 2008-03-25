#!/bin/bash
filename=`basename $1`
pathname=`dirname $1`

tag=`grep 'CREATIONTAG' $1 | head -n 1 | sed 's/.*>\(.*\)<.*/\1/'`

file_prefix=`echo $filename | sed 's/\(.*\)_[0-9]\+\.xml/\1/'`

for f in `ls $pathname/*.xml`;
  do
  mv $f $f.dat
done;

./xmlToolsRun --luts2 --prefix=$file_prefix --path=$pathname --tag=$tag

mv $tag\_Loader.xml $pathname/

#zip -j $pathname/$file_prefix.zip $pathname/*.xml*
zip -j $pathname/$tag.zip $pathname/*.xml*
