#!/bin/tcsh 
grep elem domcount.out | awk '{sum += $2} END {print sum}'
set whst=`(grep -n "Start checking" dddreport.out | awk -F : '{print $1}')`
echo whst is $whst
set totsiz=`(wc -l dddreport.out | awk '{print $1}')`
echo totsiz is $totsiz
 set tsdif = $totsiz - $whst
tail -$tsdif dddreport.out




