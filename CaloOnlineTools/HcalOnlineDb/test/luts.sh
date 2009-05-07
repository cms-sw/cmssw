#!/bin/bash
filename=`basename $1`
pathname=`dirname $1`

tag=`grep 'CREATIONTAG' $1 | head -n 1 | sed 's/.*>\(.*\)<.*/\1/'`
#tag='CR0T_test3'

file_prefix=`echo $filename | sed 's/\(.*\)_[0-9]\+\.xml/\1/'`

for f in `ls $pathname/*.xml`;
  do
  mv $f $f.dat
done;

./xmlToolsRun --luts2 --prefix=$file_prefix --path=$pathname --tag=$tag

mv $tag\_Loader.xml $pathname/

#zip -j $pathname/$file_prefix.zip $pathname/*.xml*
#zip -j $pathname/$tag.zip $pathname/*.xml*
zip -j ./$tag.zip $pathname/*.xml*

echo ''
echo 'LUTs are prepared for uploading to OMDS and saved in ./'$tag.zip
echo ''
echo 'REMEMBER!'
echo -n 'It is always a good idea to upload to the validation '
echo 'database first before uploading to OMDS'
echo ''
echo 'In order to upload to a database, copy ./'$tag.zip 'to'
echo 'dbvalhcal@pcuscms34.cern.ch:conditions/ (validation - first!)'
echo 'dbpp5hcal@pcuscms34.cern.ch:conditions/ (OMDS)'
echo ''
echo -n 'or, even better, follow the most recent instructions at '
echo 'https://twiki.cern.ch/twiki/bin/view/CMS/OnlineHCALDataSubmissionProceduresTOProdOMDSP5Server'
echo ''
