#!/bin/csh

set k=0
foreach i (`cat tmp`)
echo ${i}
#if( ${i} == "<tr>" ) then
#set k=0
#else
#if( ${k} == "1" ) then
#echo ${i}
#else
#@ k = ${k} + "1"
#endif
#endif
end
