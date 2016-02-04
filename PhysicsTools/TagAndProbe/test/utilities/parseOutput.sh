#!/bin/bash
grep "LINESHAPE EFFICIENCY:" *.out > tmp.txt
cat tmp.txt | sed "s|.out:LINESHAPE EFFICIENCY||g" > parsed_lineshape.txt
grep "qMisID EFFICIENCY:" *.out > tmp.txt
cat tmp.txt | sed "s|.out:qMisID EFFICIENCY||g" > parsed_qmisid.txt
grep "LINESHAPE EFFICIENCY:" *.out.mc > tmp.txt
cat tmp.txt | sed "s|.out.mc:LINESHAPE EFFICIENCY||g" > parsed_lineshape.txt.mc
grep "qMisID EFFICIENCY:" *.out.mc > tmp.txt
cat tmp.txt | sed "s|.out.mc:qMisID EFFICIENCY||g" > parsed_qmisid.txt.mc
grep "LINESHAPE EFFICIENCY:" *.out.mc.truth > tmp.txt
cat tmp.txt | sed "s|.out.mc.truth:LINESHAPE EFFICIENCY||g" > parsed_lineshape.txt.mc.truth
grep "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!   pass: chi2/dof=" *.out >tmp.txt
cat tmp.txt | sed "s|.out:!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!   pass: chi2/dof=||g" > parsed_passing_chi2.txt
grep "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!   fail: chi2/dof=" *.out >tmp.txt
cat tmp.txt | sed "s|.out:!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!   fail: chi2/dof=||g" > parsed_failing_chi2.txt


grep "(+" *.out > tmp.txt
cat tmp.txt | sed "s|.out:Floating Parameter  InitialValue    FinalValue (+HiError,-LoError)    GblCorr.||g" > parsed_minos0.txt
grep "LINESHAPE SIGNAL" *.out > tmp.txt
cat tmp.txt | sed "s|.out:LINESHAPE SIGNAL pass, fail=||g" > parsed_nSignal.txt
grep "LINESHAPE BKG" *.out > tmp.txt
cat tmp.txt | sed "s|.out:LINESHAPE BKG pass, fail=||g" > parsed_nBkg.txt


grep "LINESHAPE SIGNAL" *.out.mc > tmp.txt
cat tmp.txt | sed "s|.out.mc:LINESHAPE SIGNAL pass, fail=||g" > parsed_nSignal.txt.mc

grep "LINESHAPE SIGNAL" *.out.mc.truth > tmp.txt
cat tmp.txt | sed "s|.out.mc.truth:LINESHAPE SIGNAL pass, fail=||g" > parsed_nSignal.txt.mc.truth


### OK now we have all effs dumped to the txt files -- from here we want to produce a table
arrayLength=35
effArray_type[$arrayLength]=0
effArray_lineshape[$arrayLength]=0
effArray_qmisid[$arrayLength]=0
effArray_lineshape_mc[$arrayLength]=0
effArray_qmisid_mc[$arrayLength]=0
effArray_lineshape_mc_truth[$arrayLength]=0
effArray_lineshape_stat[$arrayLength]=0
effArray_qmisid_stat[$arrayLength]=0
effArray_lineshape_syst[$arrayLength]=0
effArray_qmisid_syst[$arrayLength]=0
effArray_qmisid_mc_syst[$arrayLength]=0

passChi2[$arrayLength]=0
failChi2[$arrayLength]=0

passSig[$arrayLength]=0
failSig[$arrayLength]=0
passBkg[$arrayLength]=0
failBkg[$arrayLength]=0

passSig_mc[$arrayLength]=0
failSig_mc[$arrayLength]=0

passSig_mc_truth[$arrayLength]=0
failSig_mc_truth[$arrayLength]=0



i=0
exec<parsed_lineshape.txt
while read line; do
echo $line > tmp0.txt
effArray_type[$i]=`awk '{print $1}' tmp0.txt`
effArray_lineshape[$i]=`awk '{print $2}' tmp0.txt`
effArray_lineshape_stat[$i]=`awk '{print $4}' tmp0.txt`
#echo ${effArray_type[$i][0]} ${effArray_lineshape[$i][0]} ${effArray_lineshape_stat[$i]} 
i=$(($i+1))
done


i=0
exec<parsed_lineshape.txt.mc
while read line; do
echo $line > tmp0.txt
effArray_lineshape_mc[$i]=`awk '{print $2}' tmp0.txt`
i=$(($i+1))
done

i=0
exec<parsed_lineshape.txt.mc.truth
while read line; do
echo $line > tmp0.txt
effArray_lineshape_mc_truth[$i]=`awk '{print $2}' tmp0.txt`
i=$(($i+1))
done


i=0
exec<parsed_qmisid.txt
while read line; do
echo $line > tmp0.txt
effArray_qmisid[$i]=`awk '{print $2}' tmp0.txt`
effArray_qmisid_stat[$i]=`awk '{print $4}' tmp0.txt`
effArray_qmisid_syst[$i]=`awk '{print $6}' tmp0.txt`
i=$(($i+1))
done

i=0
exec<parsed_qmisid.txt.mc
while read line; do
echo $line > tmp0.txt
effArray_qmisid_mc[$i]=`awk '{print $2}' tmp0.txt`
effArray_qmisid_mc_syst[$i]=`awk '{print $6}' tmp0.txt`
i=$(($i+1))
done


i=0
exec<parsed_passing_chi2.txt
while read line; do
echo $line > tmp0.txt
passChi2[$i]=`awk '{print $2}' tmp0.txt`
i=$(($i+1))
done

i=0
exec<parsed_failing_chi2.txt
while read line; do
echo $line > tmp0.txt
failChi2[$i]=`awk '{print $2}' tmp0.txt`
i=$(($i+1))
done

i=0
exec<parsed_nSignal.txt
while read line; do
echo $line > tmp0.txt
passSig[$i]=`awk '{print $2}' tmp0.txt`
failSig[$i]=`awk '{print $5}' tmp0.txt`
i=$(($i+1))
done

i=0
exec<parsed_nBkg.txt
while read line; do
echo $line > tmp0.txt
passBkg[$i]=`awk '{print $2}' tmp0.txt`
failBkg[$i]=`awk '{print $5}' tmp0.txt`
i=$(($i+1))
done


i=0
exec<parsed_nSignal.txt.mc
while read line; do
echo $line > tmp0.txt
passSig_mc[$i]=`awk '{print $2}' tmp0.txt`
failSig_mc[$i]=`awk '{print $5}' tmp0.txt`
i=$(($i+1))
done

i=0
exec<parsed_nSignal.txt.mc.truth
while read line; do
echo $line > tmp0.txt
passSig_mc_truth[$i]=`awk '{print $2}' tmp0.txt`
failSig_mc_truth[$i]=`awk '{print $5}' tmp0.txt`
i=$(($i+1))
done





### OK -- now go all info we need
mv efficiencyTable.txt efficiencyTable_old.txt
echo "type                         line           line_stat        mc       mc_truth        qmisid         qmisid_stat      qmisid_syst      qmisid_mc      qmisid_mc_syst" >> efficiencyTable.txt
i=0
while [ $i -lt $arrayLength ]; do
foo=${#effArray_type[$i]}
add=$((25-$foo)) 
x=""
j=0
#echo add is $add
while [ $j -lt $add ]; do
#x=`echo "$x + 1" | bc`
x=`echo "$x-"`
#echo x is $x
j=$(($j+1))
done
x=`echo "$x :"`
if [[ ${effArray_type[$i]} = *gsf* ]]; then
    echo ${effArray_type[$i]}  $x    ${effArray_lineshape[$i]} "     "     ${effArray_lineshape_stat[$i]} "     "     ${effArray_lineshape_mc[$i]} "     "     ${effArray_lineshape_mc_truth[$i]} >> efficiencyTable.txt
elif [[ ${effArray_type[$i]} = *plus* ]]; then
echo ${effArray_type[$i]}  $x    ${effArray_lineshape[$i]} "     "     ${effArray_lineshape_stat[$i]} "     "     ${effArray_lineshape_mc[$i]} "     "     ${effArray_lineshape_mc_truth[$i]}   >> efficiencyTable.txt
elif [[ ${effArray_type[$i]} = *minus* ]]; then
echo ${effArray_type[$i]}  $x    ${effArray_lineshape[$i]} "     "     ${effArray_lineshape_stat[$i]} "     "     ${effArray_lineshape_mc[$i]} "     "     ${effArray_lineshape_mc_truth[$i]}   >> efficiencyTable.txt
else 
echo ${effArray_type[$i]}  $x    ${effArray_lineshape[$i]} "     "     ${effArray_lineshape_stat[$i]} "     "     ${effArray_lineshape_mc[$i]} "     "     ${effArray_lineshape_mc_truth[$i]} "     "     ${effArray_qmisid[$i]} "     "    ${effArray_qmisid_stat[$i]} "     "     ${effArray_qmisid_syst[$i]} "     "     ${effArray_qmisid_mc[$i]} "     "     ${effArray_qmisid_mc_syst[$i]}   >> efficiencyTable.txt
fi
i=$(($i+1))  
done





mv efficiencyTable_MC.txt efficiencyTable_MC_old.txt
echo "type                         mc             mc_truth       mc_pass_integral         mc_fail_integral      mc_truth_pass_integral      mc_truth_fail_integral" >> efficiencyTable_MC.txt
i=0
while [ $i -lt $arrayLength ]; do
foo=${#effArray_type[$i]}
add=$((25-$foo)) 
x=""
j=0
#echo add is $add
while [ $j -lt $add ]; do
#x=`echo "$x + 1" | bc`
x=`echo "$x-"`
#echo x is $x
j=$(($j+1))
done
x=`echo "$x :"`
echo ${effArray_type[$i]}  $x    ${effArray_lineshape_mc[$i]} "     "     ${effArray_lineshape_mc_truth[$i]} "     "     ${passSig_mc[$i]} "                 "    ${failSig_mc[$i]} "                "     ${passSig_mc_truth[$i]} "                    "     ${failSig_mc_truth[$i]}   >> efficiencyTable_MC.txt

i=$(($i+1))  
done






mv detailedTable.txt detailedTable_old.txt
echo "type                        eff             stat             chi2_pass      chi2_fail      sig_pass      bkg_pass      sig_fail      bkg_fail    fVars_pass      fvars_errors_pass           fVars_fail       fVars_errors_fail" > detailedTable.txt
i=0
echo "arrayLength= " $arrayLength
#while [ $i -lt 7 ]; do
while [ $i -lt $arrayLength ]; do
foo=${#effArray_type[$i]}
add=$((25-$foo))
x=""
j=0
while [ $j -lt $add ]; do
x=`echo "$x-"`
j=$(($j+1))
done
x=`echo "$x :"`
echo -n ${effArray_type[$i]}  $x    ${effArray_lineshape[$i]} "     "     ${effArray_lineshape_stat[$i]} "     "  ${passChi2[$i]}   "     "  ${failChi2[$i]} "     "  ${passSig[$i]}   "     "  ${passBkg[$i]} "     "  ${failSig[$i]}   "     "  ${failBkg[$i]} >> detailedTable.txt

###now throw in the minos errors for the floating variables
pass_par[5]=0
pass_err[5]=0
fail_par[5]=0
fail_err[5]=0

pass_par[0]=" "
pass_err[0]=" "
fail_par[0]=" "
fail_err[0]=" "

pass_par[1]=" "
pass_err[1]=" "
fail_par[1]=" "
fail_err[1]=" "

pass_par[2]=" "
pass_err[2]=" "
fail_par[2]=" "
fail_err[2]=" "

pass_par[3]=" "
pass_err[3]=" "
fail_par[3]=" "
fail_err[3]=" "

pass_par[4]=" "
pass_err[4]=" "
fail_par[4]=" "
fail_err[4]=" "


nPassPar=0
nFailPar=0
j=0
exec<parsed_minos0.txt
while read line; do
#echo $line > tmp1.txt
echo $line | sed "s|.out||g"  > tmp0.txt
type=`awk '{print $1}' tmp0.txt`
#echo $type ${effArray_type[$i]}
if [[ $type = ${effArray_type[$i]} ]]; then
    what=`awk '{print $2}' tmp0.txt`
    #echo $what $j
    if [[ $what != *Float* ]]; then
	
	if [[ $j -eq 1 ]]; then 
	    pass_err[$nPassPar]=`awk '{print $5}' tmp0.txt`
	    pass_par[$nPassPar]=`awk '{print $2}' tmp0.txt`
	    nPassPar=$(($nPassPar+1)) 
	    #echo "yeah" $nPassPar
	elif [[ $j -eq 2 ]]; then
            fail_err[$nFailPar]=`awk '{print $5}' tmp0.txt`
            fail_par[$nFailPar]=`awk '{print $2}' tmp0.txt`
            nFailPar=$(($nFailPar+1))
	    #echo "yeahF" $nFailPar
	fi
    else
	j=$(($j+1))
    fi
fi    
###now finally print the minos errors for the floating variables
done
j=0
#echo "npasspar failpar= " $nPassPar " " $nFailPar
if [ $nPassPar -gt $nFailPar ]; then
    echo "    " ${pass_par[0]} "         " ${pass_err[0]}   "     "  ${fail_par[0]}   "            "    ${fail_err[0]}    >>  detailedTable.txt
    j=$(($j+1))
    while [ $j -lt $nPassPar ]; do  
	echo "                                                                                                                                                " ${pass_par[$j]} "         " ${pass_err[$j]}   "     "  ${fail_par[$j]}   "            "    ${fail_err[$j]}    >>  detailedTable.txt
	j=$(($j+1))
    done
else
    echo "    " ${pass_par[0]} "         " ${pass_err[0]}   "     "  ${fail_par[0]}   "            "    ${fail_err[0]}    >>  detailedTable.txt
    j=$(($j+1))
    while [ $j -lt $nFailPar ];do
        echo "                                                                                                                                                " ${pass_par[$j]} "         " ${pass_err[$j]}   "      "  ${fail_par[$j]}   "            "    ${fail_err[$j]}    >>  detailedTable.txt
	j=$(($j+1))
    done
fi    
i=$(($i+1))  
echo "i= " $i
done

#get rid of pesky commas
#mv detailedTable.txt tmp.txt
#cat tmp.txt | sed "s|,||g" >detailedTable.txt