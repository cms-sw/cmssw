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
#set setName=btagMuJetsWpNoTtbar
#set setName=btagMistagWinter13
#set setName=btagMistagAB
#set setName=btagMistagC
#set setName=btagMistagD


#Unique version number for DB
set version=v10
cat templates/Pool_pre.fragment | sed "s#SETNAME#$setName#g"  > Pool_$setName.py
cat templates/Btag_pre.fragment > Btag_$setName.py

#set tag=PerformancePayloadFromTable
set tag=PerformancePayloadFromBinnedTFormula

#"mistag" measurements go here



#Create a single measurement with ./makeSingle.csh <file path> <measurement name> <set name>
#./makeSingle.csh BTAG/FromLocal/mujets_wp/ttbar/BTAGTTBARCSVL.txt       MUJETSWPBTAGTTBARCSVL $setName $version $tag
#./makeSingle.csh BTAG/FromLocal/mujets_wp/ttbar/BTAGTTBARCSVM.txt       MUJETSWPBTAGTTBARCSVM $setName $version $tag
#./makeSingle.csh BTAG/FromLocal/mujets_wp/ttbar/BTAGTTBARCSVT.txt       MUJETSWPBTAGTTBARCSVT $setName $version $tag
#./makeSingle.csh BTAG/FromLocal/mujets_wp/ttbar/BTAGTTBARCSVV1L.txt     MUJETSWPBTAGTTBARCSVV1L $setName $version $tag
#./makeSingle.csh BTAG/FromLocal/mujets_wp/ttbar/BTAGTTBARCSVV1M.txt     MUJETSWPBTAGTTBARCSVV1M $setName $version $tag
#./makeSingle.csh BTAG/FromLocal/mujets_wp/ttbar/BTAGTTBARCSVV1T.txt     MUJETSWPBTAGTTBARCSVV1T $setName $version $tag
#./makeSingle.csh BTAG/FromLocal/mujets_wp/ttbar/BTAGTTBARCSVSLV1L.txt   MUJETSWPBTAGTTBARCSVSLV1L $setName $version $tag
#./makeSingle.csh BTAG/FromLocal/mujets_wp/ttbar/BTAGTTBARCSVSLV1M.txt   MUJETSWPBTAGTTBARCSVSLV1M $setName $version $tag
#./makeSingle.csh BTAG/FromLocal/mujets_wp/ttbar/BTAGTTBARCSVSLV1T.txt   MUJETSWPBTAGTTBARCSVSLV1T $setName $version $tag
#./makeSingle.csh BTAG/FromLocal/mujets_wp/ttbar/BTAGTTBARTCHPT.txt      MUJETSWPBTAGTTBARTCHPT $setName $version $tag

#./makeSingle.csh BTAG/FromLocal/mujets_wp/NOttbar/BTAGNOTTBARCSVL.txt       MUJETSWPBTAGNOTTBARCSVL $setName $version $tag
#./makeSingle.csh BTAG/FromLocal/mujets_wp/NOttbar/BTAGNOTTBARCSVM.txt       MUJETSWPBTAGNOTTBARCSVM $setName $version $tag
#./makeSingle.csh BTAG/FromLocal/mujets_wp/NOttbar/BTAGNOTTBARCSVT.txt       MUJETSWPBTAGNOTTBARCSVT $setName $version $tag
#./makeSingle.csh BTAG/FromLocal/mujets_wp/NOttbar/BTAGNOTTBARCSVV1L.txt     MUJETSWPBTAGNOTTBARCSVV1L $setName $version $tag
#./makeSingle.csh BTAG/FromLocal/mujets_wp/NOttbar/BTAGNOTTBARCSVV1M.txt     MUJETSWPBTAGNOTTBARCSVV1M $setName $version $tag
#./makeSingle.csh BTAG/FromLocal/mujets_wp/NOttbar/BTAGNOTTBARCSVV1T.txt     MUJETSWPBTAGNOTTBARCSVV1T $setName $version $tag#
#./makeSingle.csh BTAG/FromLocal/mujets_wp/NOttbar/BTAGNOTTBARCSVSLV1L.txt   MUJETSWPBTAGNOTTBARCSVSLV1L $setName $version $tag
#./makeSingle.csh BTAG/FromLocal/mujets_wp/NOttbar/BTAGNOTTBARCSVSLV1M.txt   MUJETSWPBTAGNOTTBARCSVSLV1M $setName $version $tag
#./makeSingle.csh BTAG/FromLocal/mujets_wp/NOttbar/BTAGNOTTBARCSVSLV1T.txt   MUJETSWPBTAGNOTTBARCSVSLV1T $setName $version $tag
#./makeSingle.csh BTAG/FromLocal/mujets_wp/NOttbar/BTAGNOTTBARJPL.txt        MUJETSWPBTAGNOTTBARJPL $setName $version $tag
#./makeSingle.csh BTAG/FromLocal/mujets_wp/NOttbar/BTAGNOTTBARJPM.txt        MUJETSWPBTAGNOTTBARJPM $setName $version $tag
#./makeSingle.csh BTAG/FromLocal/mujets_wp/NOttbar/BTAGNOTTBARJPT.txt        MUJETSWPBTAGNOTTBARJPT $setName $version $tag
#./makeSingle.csh BTAG/FromLocal/mujets_wp/NOttbar/BTAGNOTTBARTCHPT.txt      MUJETSWPBTAGNOTTBARTCHPT $setName $version $tag


#./makeSingle.csh BTAG/FromLocal/mujets_wp/ttbar/BTAGCSVL.txt MUJETSWPBTAGCSVLTTBAR $setName $version $tag
#./makeSingle.csh BTAG/FromLocal/mujets_wp/ttbar/BTAGCSVM.txt MUJETSWPBTAGCSVMTTBAR $setName $version $tag
#./makeSingle.csh BTAG/FromLocal/mujets_wp/ttbar/BTAGCSVT.txt MUJETSWPBTAGCSVTTTBAR $setName $version $tag
#./makeSingle.csh BTAG/FromLocal/mujets_wp/ttbar/BTAGJPL.txt MUJETSWPBTAGJPLTTBAR $setName $version $tag
#./makeSingle.csh BTAG/FromLocal/mujets_wp/ttbar/BTAGJPM.txt MUJETSWPBTAGJPMTTBAR $setName $version $tag
#./makeSingle.csh BTAG/FromLocal/mujets_wp/ttbar/BTAGJPT.txt MUJETSWPBTAGJPTTTBAR $setName $version $tag
#./makeSingle.csh BTAG/FromLocal/mujets_wp/ttbar/BTAGTCHPT.txt MUJETSWPBTAGTCHPT $setName $version $tag

./makeSingle.csh BTAG/FromLocal/TTbar/BTAGCSV.txt TTBARDISCRIMBTAGCSV $setName $version $tag
./makeSingle.csh BTAG/FromLocal/TTbar/BTAGJP.txt TTBARDISCRIMBTAGJP $setName $version $tag
./makeSingle.csh BTAG/FromLocal/TTbar/BTAGTCHP.txt TTBARDISCRIMBTAGTCHP $setName $version $tag


####################
###
### make sure to change write_template.py for WP's and not formula
###
###################

#./makeSingle.csh BTAG/FromLocal/SFLIGHT/MISTAGCSVL.txt MISTAGCSVL $setName $version $tag
#./makeSingle.csh BTAG/FromLocal/SFLIGHT/MISTAGCSVM.txt MISTAGCSVM $setName $version $tag
#./makeSingle.csh BTAG/FromLocal/SFLIGHT/MISTAGCSVT.txt MISTAGCSVT $setName $version $tag
#./makeSingle.csh BTAG/FromLocal/SFLIGHT/MISTAGCSVSLV1L.txt MISTAGCSVSLV1L $setName $version $tag
#./makeSingle.csh BTAG/FromLocal/SFLIGHT/MISTAGCSVSLV1M.txt MISTAGCSVSLV1M $setName $version $tag
#./makeSingle.csh BTAG/FromLocal/SFLIGHT/MISTAGCSVSLV1T.txt MISTAGCSVSLV1T $setName $version $tag
#./makeSingle.csh BTAG/FromLocal/SFLIGHT/MISTAGCSVV1L.txt MISTAGCSVV1L $setName $version $tag
#./makeSingle.csh BTAG/FromLocal/SFLIGHT/MISTAGCSVV1M.txt MISTAGCSVV1M $setName $version $tag
#./makeSingle.csh BTAG/FromLocal/SFLIGHT/MISTAGCSVV1T.txt MISTAGCSVV1T $setName $version $tag
#./makeSingle.csh BTAG/FromLocal/SFLIGHT/MISTAGJPL.txt MISTAGJPL $setName $version $tag
#./makeSingle.csh BTAG/FromLocal/SFLIGHT/MISTAGJPM.txt MISTAGJPM $setName $version $tag
#./makeSingle.csh BTAG/FromLocal/SFLIGHT/MISTAGJPT.txt MISTAGJPT $setName $version $tag
#./makeSingle.csh BTAG/FromLocal/SFLIGHT/MISTAGTCHPT.txt MISTAGTCHPT $setName $version $tag

#./makeSingle.csh BTAG/FromLocal/TTbar/BTAGCSVL.txt TTBARWPBTAGCSVL $setName $version $tag
#./makeSingle.csh BTAG/FromLocal/TTbar/BTAGCSVM.txt TTBARWPBTAGCSVM $setName $version $tag
#./makeSingle.csh BTAG/FromLocal/TTbar/BTAGCSVT.txt TTBARWPBTAGCSVT $setName $version $tag
#./makeSingle.csh BTAG/FromLocal/TTbar/BTAGJPL.txt TTBARWPBTAGJPL $setName $version $tag
#./makeSingle.csh BTAG/FromLocal/TTbar/BTAGJPM.txt TTBARWPBTAGJPM $setName $version $tag
#./makeSingle.csh BTAG/FromLocal/TTbar/BTAGJPT.txt TTBARWPBTAGJPT $setName $version $tag
#./makeSingle.csh BTAG/FromLocal/TTbar/BTAGTCHPT.txt TTBARWPBTAGTCHPT $setName $version $tag


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



