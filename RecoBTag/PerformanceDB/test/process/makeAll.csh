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
#set setName=btagTtbarWp
set setName=btagTtbarDiscrim
#set setName=btagMuJetsWp
#set setName=btagMistagABCD
#set setName=btagMistagAB
#set setName=btagMistagC
#set setName=btagMistagD


#Unique version number for DB
set version=v9
cat templates/Pool_pre.fragment | sed "s#SETNAME#$setName#g"  > Pool_$setName.py
cat templates/Btag_pre.fragment > Btag_$setName.py

#set tag=PerformancePayloadFromTable
set tag=PerformancePayloadFromBinnedTFormula

#"mistag" measurements go here



#Create a single measurement with ./makeSingle.csh <file path> <measurement name> <set name>
# ./makeSingle.csh BTAG/mujets_wp/BTAGCSVL.txt MUJETSWPBTAGCSVL $setName $version $tag
# ./makeSingle.csh BTAG/mujets_wp/BTAGCSVM.txt MUJETSWPBTAGCSVM $setName $version $tag
# ./makeSingle.csh BTAG/mujets_wp/BTAGCSVT.txt MUJETSWPBTAGCSVT $setName $version $tag
# ./makeSingle.csh BTAG/mujets_wp/BTAGJPL.txt MUJETSWPBTAGJPL $setName $version $tag
# ./makeSingle.csh BTAG/mujets_wp/BTAGJPM.txt MUJETSWPBTAGJPM $setName $version $tag
# ./makeSingle.csh BTAG/mujets_wp/BTAGJPT.txt MUJETSWPBTAGJPT $setName $version $tag
# ./makeSingle.csh BTAG/mujets_wp/BTAGTCHPT.txt MUJETSWPBTAGTCHPT $setName $version $tag

 ./makeSingle.csh BTAG/ttbar/BTAGCSV.txt TTBARDISCRIMBTAGCSV $setName $version $tag
 ./makeSingle.csh BTAG/ttbar/BTAGJP.txt TTBARDISCRIMBTAGJP $setName $version $tag
 ./makeSingle.csh BTAG/ttbar/BTAGTCHP.txt TTBARDISCRIMBTAGTCHP $setName $version $tag


####################
###
### make sure to change write_template.py for WP's and not formula
###
###################

# ./makeSingle.csh BTAG/ttbar_wp/BTAGCSVL.txt TTBARWPBTAGCSVL $setName $version $tag
# ./makeSingle.csh BTAG/ttbar_wp/BTAGCSVM.txt TTBARWPBTAGCSVM $setName $version $tag
# ./makeSingle.csh BTAG/ttbar_wp/BTAGCSVT.txt TTBARWPBTAGCSVT $setName $version $tag
# ./makeSingle.csh BTAG/ttbar_wp/BTAGJPL.txt TTBARWPBTAGJPL $setName $version $tag
# ./makeSingle.csh BTAG/ttbar_wp/BTAGJPM.txt TTBARWPBTAGJPM $setName $version $tag
# ./makeSingle.csh BTAG/ttbar_wp/BTAGJPT.txt TTBARWPBTAGJPT $setName $version $tag
# ./makeSingle.csh BTAG/ttbar_wp/BTAGTCHPT.txt TTBARWPBTAGTCHPT $setName $version $tag


# cat templates/Pool_post.fragment | sed "s#SETNAME#$setName#g" >> Pool_$setName.py

# set setName=mistag

# cat templates/Pool_pre.fragment | sed "s#SETNAME#$setName#g"  > Pool_$setName.py
# cat templates/Btag_pre.fragment > Btag_$setName.py

# ./makeSingle.csh BTAG/SFlight/DataPeriod_ABCD/MISTAGCSVL.txt MISTAGCSVLABCD $setName $version $tag
# ./makeSingle.csh BTAG/SFlight/DataPeriod_ABCD/MISTAGCSVM.txt MISTAGCSVMABCD $setName $version $tag
# ./makeSingle.csh BTAG/SFlight/DataPeriod_ABCD/MISTAGCSVT.txt MISTAGCSVTABCD $setName $version $tag
# ./makeSingle.csh BTAG/SFlight/DataPeriod_ABCD/MISTAGJPL.txt  MISTAGJPLABCD $setName $version $tag
# ./makeSingle.csh BTAG/SFlight/DataPeriod_ABCD/MISTAGJPM.txt  MISTAGJPMABCD $setName $version $tag
# ./makeSingle.csh BTAG/SFlight/DataPeriod_ABCD/MISTAGJPT.txt  MISTAGJPTABCD $setName $version $tag
# ./makeSingle.csh BTAG/SFlight/DataPeriod_ABCD/MISTAGTCHPT.txt MISTAGTCHPTABCD $setName $version $tag

# ./makeSingle.csh BTAG/SFlight/DataPeriod_AB/MISTAGCSVL.txt MISTAGCSVLAB $setName $version $tag
# ./makeSingle.csh BTAG/SFlight/DataPeriod_AB/MISTAGCSVM.txt MISTAGCSVMAB $setName $version $tag
# ./makeSingle.csh BTAG/SFlight/DataPeriod_AB/MISTAGCSVT.txt MISTAGCSVTAB $setName $version $tag
# ./makeSingle.csh BTAG/SFlight/DataPeriod_AB/MISTAGJPL.txt  MISTAGJPLAB $setName $version $tag
# ./makeSingle.csh BTAG/SFlight/DataPeriod_AB/MISTAGJPM.txt  MISTAGJPMAB $setName $version $tag
# ./makeSingle.csh BTAG/SFlight/DataPeriod_AB/MISTAGJPT.txt  MISTAGJPTAB $setName $version $tag
# ./makeSingle.csh BTAG/SFlight/DataPeriod_AB/MISTAGTCHPT.txt MISTAGTCHPTAB $setName $version $tag

# ./makeSingle.csh BTAG/SFlight/DataPeriod_C/MISTAGCSVL.txt MISTAGCSVLC $setName $version $tag
# ./makeSingle.csh BTAG/SFlight/DataPeriod_C/MISTAGCSVM.txt MISTAGCSVMC $setName $version $tag
# ./makeSingle.csh BTAG/SFlight/DataPeriod_C/MISTAGCSVT.txt MISTAGCSVTC $setName $version $tag
# ./makeSingle.csh BTAG/SFlight/DataPeriod_C/MISTAGJPL.txt  MISTAGJPLC $setName $version $tag
# ./makeSingle.csh BTAG/SFlight/DataPeriod_C/MISTAGJPM.txt  MISTAGJPMC $setName $version $tag
# ./makeSingle.csh BTAG/SFlight/DataPeriod_C/MISTAGJPT.txt  MISTAGJPTC $setName $version $tag
# ./makeSingle.csh BTAG/SFlight/DataPeriod_C/MISTAGTCHPT.txt MISTAGTCHPTC $setName $version $tag

# ./makeSingle.csh BTAG/SFlight/DataPeriod_D/MISTAGCSVL.txt MISTAGCSVLD $setName $version $tag
# ./makeSingle.csh BTAG/SFlight/DataPeriod_D/MISTAGCSVM.txt MISTAGCSVMD $setName $version $tag
# ./makeSingle.csh BTAG/SFlight/DataPeriod_D/MISTAGCSVT.txt MISTAGCSVTD $setName $version $tag
# ./makeSingle.csh BTAG/SFlight/DataPeriod_D/MISTAGJPL.txt  MISTAGJPLD $setName $version $tag
# ./makeSingle.csh BTAG/SFlight/DataPeriod_D/MISTAGJPM.txt  MISTAGJPMD $setName $version $tag
# ./makeSingle.csh BTAG/SFlight/DataPeriod_D/MISTAGJPT.txt  MISTAGJPTD $setName $version $tag
# ./makeSingle.csh BTAG/SFlight/DataPeriod_D/MISTAGTCHPT.txt MISTAGTCHPTD $setName $version $tag





cat templates/Pool_post.fragment | sed "s#SETNAME#$setName#g" >> Pool_$setName.py



