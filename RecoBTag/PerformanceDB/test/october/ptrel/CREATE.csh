#!/bin/tcsh
eval `scramv1 runtime -csh`
set name=$1
set input=write_template.py
set inputpoolfrag=../Pool_template.fragment
set inputbtagfragment=../Btag_template.fragment
set oututfragname=Pool_template.$name
set oututbtagfragname=Btag_template.$name
set taggers=(`cat WPs`)

#rm -f PhysicsPerformance.db

rm -f $oututfragname
touch $oututfragname
rm -f $oututbtagfragname
touch $oututbtagfragname

foreach i (`echo $taggers`)
    echo $i
    rm -f tmp.py
    cat $input | sed "s/TEMPLATE/$i/g" |sed "s/NAME/$name/g"> tmp.py
    cat $inputpoolfrag | sed "s/T1/$name/g" |sed "s/T2/$i/g" >> $oututfragname 
    cat $inputbtagfragment |sed "s/TEMPLATE/$name$i/g" >> $oututbtagfragname
    cmsRun tmp.py
end
#
