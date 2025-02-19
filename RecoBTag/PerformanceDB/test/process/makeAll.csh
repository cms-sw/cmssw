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
set setName=btagTtbarWp
#Unique version number for DB
set version=v8
cat templates/Pool_pre.fragment | sed "s#SETNAME#$setName#g"  > Pool_$setName.py
cat templates/Btag_pre.fragment > Btag_$setName.py

#"mistag" measurements go here
#Create a single measurement with ./makeSingle.csh <file path> <measurement name> <set name>
 ./makeSingle.csh BTAG/ttbar_wp/BTAGCSVL.txt BTAGCSVL $setName $version
 ./makeSingle.csh BTAG/ttbar_wp/BTAGCSVM.txt BTAGCSVM $setName $version
 ./makeSingle.csh BTAG/ttbar_wp/BTAGCSVT.txt BTAGCSVT $setName $version
# ./makeSingle.csh BTAG/BTAGJBPL.txt BTAGJBPL $setName $version
# ./makeSingle.csh BTAG/BTAGJBPM.txt BTAGJBPM $setName $version
# ./makeSingle.csh BTAG/BTAGJBPT.txt BTAGJBPT $setName $version
# ./makeSingle.csh BTAG/BTAGJPL.txt BTAGJPL $setName $version
# ./makeSingle.csh BTAG/BTAGJPM.txt BTAGJPM $setName $version
# ./makeSingle.csh BTAG/BTAGJPT.txt BTAGJPT $setName $version
 ./makeSingle.csh BTAG/ttbar_wp/BTAGSSVHEM.txt BTAGSSVHEM $setName $version
 ./makeSingle.csh BTAG/ttbar_wp/BTAGSSVHPT.txt BTAGSSVHPT $setName $version
 ./makeSingle.csh BTAG/ttbar_wp/BTAGTCHEL.txt BTAGTCHEL $setName $version
./makeSingle.csh BTAG/ttbar_wp/BTAGTCHEM.txt BTAGTCHEM $setName $version
./makeSingle.csh BTAG/ttbar_wp/BTAGTCHET.txt BTAGTCHET $setName $version
./makeSingle.csh BTAG/ttbar_wp/BTAGTCHPL.txt BTAGTCHPL $setName $version
./makeSingle.csh BTAG/ttbar_wp/BTAGTCHPM.txt BTAGTCHPM $setName $version
 ./makeSingle.csh BTAG/ttbar_wp/BTAGTCHPT.txt BTAGTCHPT $setName $version

# cat templates/Pool_post.fragment | sed "s#SETNAME#$setName#g" >> Pool_$setName.py

# set setName=mistag

# cat templates/Pool_pre.fragment | sed "s#SETNAME#$setName#g"  > Pool_$setName.py
# cat templates/Btag_pre.fragment > Btag_$setName.py

# ./makeSingle.csh MISTAG/MISTAGCSVLtable.txt MISTAGCSVL $setName $version
# ./makeSingle.csh MISTAG/MISTAGCSVMtable.txt MISTAGCSVM $setName $version
# ./makeSingle.csh MISTAG/MISTAGCSVTtable.txt MISTAGCSVT $setName $version
# ./makeSingle.csh MISTAG/MISTAGJBPLtable.txt MISTAGJBPL $setName $version
# ./makeSingle.csh MISTAG/MISTAGJBPMtable.txt MISTAGJBPM $setName $version
# ./makeSingle.csh MISTAG/MISTAGJBPTtable.txt MISTAGJBPT $setName $version
# ./makeSingle.csh MISTAG/MISTAGJPLtable.txt MISTAGJPL $setName $version
# ./makeSingle.csh MISTAG/MISTAGJPMtable.txt MISTAGJPM $setName $version
# ./makeSingle.csh MISTAG/MISTAGJPTtable.txt MISTAGJPT $setName $version
# ./makeSingle.csh MISTAG/MISTAGSSVHEMtable.txt MISTAGSSVHEM $setName $version
# ./makeSingle.csh MISTAG/MISTAGSSVHPTtable.txt MISTAGSSVHPT $setName $version
# ./makeSingle.csh MISTAG/MISTAGTCHELtable.txt MISTAGTCHEL $setName $version
# ./makeSingle.csh MISTAG/MISTAGTCHEMtable.txt MISTAGTCHEM $setName $version
# ./makeSingle.csh MISTAG/MISTAGTCHPMtable.txt MISTAGTCHPM $setName $version
# ./makeSingle.csh MISTAG/MISTAGTCHPTtable.txt MISTAGTCHPT $setName $version


cat templates/Pool_post.fragment | sed "s#SETNAME#$setName#g" >> Pool_$setName.py
