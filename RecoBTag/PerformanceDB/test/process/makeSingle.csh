#!/bin/tcsh
eval `scramv1 runtime -csh`

set name=$2
set file=$1
set input=templates/write_template.py
set inputpoolfrag=templates/Pool_template.py
set inputbtagfragment=templates/BTag_template.py
set inputtestfragment=templates/testDB.py
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

cat templates/Pool_template.fragment | sed "s#TEMPLATE#$name#g" >> Pool_template.py
cat templates/Btag_template.fragment | sed "s#TEMPLATE#$name#g" >> Btag_template.py
rm -f DBs/$name.db
cmsRun tmp.py
