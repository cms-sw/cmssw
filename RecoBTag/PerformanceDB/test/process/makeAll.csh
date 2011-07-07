#!/bin/tcsh

set inputpoolheader=templates/Pool_header.fragment
set inputbtagheader=templates/Btag_header.fragment
set inputtestheader=templates/Test_header.fragment

set inputpoolheader=templates/Pool_footer.fragment
#set inputbtagheader=templates/Btag_footer.fragment
set inputtestheader=templates/Test_footer.fragment

set outputfragname=Pool_template.py
set outputbtagfragname=Btag_template.py
set outputtestfragname=Test_template.py


#Make our directories in order!
mkdir DBs
mkdir ship
mkdir -p testOnline/text
mkdir -p test/text

#Remove any remaining db making code
rm -f tmp.py

#For a set of measurements we want a unique name
set setName=mistag
#Unique version number for DB
set version=6
cat templates/Pool_pre.fragment | sed "s#SETNAME#$setName#g"  > Pool_$setName.py
cat templates/Btag_pre.fragment > Btag_$setName.py

#"mistag" measurements go here
#Create a single measurement with ./makeSingle.csh <file path> <measurement name> <set name>

./makeSingle.csh MISTAG/MISTAGJPL.txt  MISTAGJPL $setName $version
./makeSingle.csh MISTAG/MISTAGJPM.txt  MISTAGJPM $setName $version
./makeSingle.csh MISTAG/MISTAGJPT.txt  MISTAGJPT $setName $version
./makeSingle.csh MISTAG/MISTAGSSVM.txt MISTAGSSVM $setName $version
./makeSingle.csh MISTAG/MISTAGTCHEL.txt MISTAGTCHEL $setName $version
./makeSingle.csh MISTAG/MISTAGTCHEM.txt MISTAGTCHEM $setName $version
./makeSingle.csh MISTAG/MISTAGTCHPM.txt MISTAGTCHPM $setName $version
./makeSingle.csh MISTAG/MISTAGTCHPT.txt MISTAGTCHPT $setName $version

cat templates/Pool_post.fragment | sed "s#SETNAME#$setName#g" >> Pool_$setName.py
