#!/bin/tcsh
eval `scramv1 runtime -csh`

set file=$1
set name=$2
set setName=$3
set version=$4
set tag=$5

set input=templates/write_template.py
set inputpoolfrag=templates/Pool_template.py
set inputbtagfragment=templates/BTag_template.py
set inputtestfragment=templates/testDB.py

#Online fragments
set inputtestfragmentOnline=templates/testDB_Online.py
set inputpoolfragOnline=templates/Pool_template_Online.py
set inputbtagfragmentOnline=templates/BTag_template_Online.py

#set oututfragname=templates/Pool_template.py
#set oututbtagfragname=templates/BTag_template.py
set outputtestfragname=Test_template.py

rm -f tmp.py

cat $input | sed  "s#FILE#$file#g" | sed  "s#NAME#$name#g"> tmp.py
rm -f test/Pool_$name.py
cat $inputpoolfrag | sed  "s#NAME#$name#g" > test/Pool_$name.py
rm -f test/Btag_$name.py
cat $inputbtagfragment | sed  "s#NAME#$name#g" > test/Btag_$name.py
rm -f test/test_$name.py
cat $inputtestfragment | sed "s#NAME#$name#g" | sed "s#FILE#$file#g" > test/test_$name.py


#Online test
rm -f testOnline/Pool_$name.py
cat $inputpoolfragOnline | sed  "s#TEMPLATE#$name#g" | sed "s#VERSION#$version#g" > testOnline/Pool_$name.py
rm -f testOnline/Btag_$name.py
cat $inputbtagfragmentOnline | sed  "s#TEMPLATE#$name#g"  | sed "s#VERSION#$version#g" > testOnline/Btag_$name.py
rm -f testOnline/test_$name.py
cat $inputtestfragmentOnline | sed "s#NAME#$name#g" | sed "s#FILE#$file#g" > testOnline/test_$name.py

cat templates/Pool_template.fragment | sed "s#TAG#$tag#g" | sed "s#TEMPLATE#$name#g" | sed "s#VERSION#$version#g" >> Pool_$setName.py
cat templates/Btag_template.fragment | sed "s#TEMPLATE#$name#g" | sed "s#VERSION#$version#g" >> Btag_$setName.py

rm -f DBs/$name.db
cmsRun tmp.py
