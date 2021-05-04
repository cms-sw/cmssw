#!/bin/tcsh
touch list2
foreach i (`cat list1`)
echo ${i} 
set size=`echo ${i} | awk -F _  '{print $1}'`
echo ${size}
if( ${size} < "1000" ) then
echo ${i} >> list2
endif
end
