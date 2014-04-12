#!/bin/bash
filename=`basename $1`
pathname=`dirname $1`
comment=$2
version=$3
subversion=$4

tag=`grep 'CREATIONTAG' $1 | head -n 1 | sed 's/.*>\(.*\)<.*/\1/'`
#tag='CR0T_test3'

file_prefix=`echo $filename | sed 's/\(.*\)_[0-9]\+\.xml/\1/'`

for f in `ls $pathname/*.xml`;
  do
  mv $f $f.dat
done;

work_dir=`pwd`
cd $pathname
#file_list=`ls $pathname/*[0-9].xml.dat`
file_list=`ls ./*[0-9].xml.dat`
cd $work_dir

#./xmlToolsRun --luts2 --prefix=$file_prefix --path=$pathname --tag=$tag
./xmlToolsRun --create-lut-loader --file-list="$file_list" --prefix-name=$file_prefix --tag-name=$tag --version-name="$version" --sub-version="$subversion" --comment-line="$comment"

mv $tag\_Loader.xml $pathname/

#zip -j $pathname/$file_prefix.zip $pathname/*.xml*
#zip -j $pathname/$tag.zip $pathname/*.xml*
zip -j ./$tag.zip $pathname/*.xml*

