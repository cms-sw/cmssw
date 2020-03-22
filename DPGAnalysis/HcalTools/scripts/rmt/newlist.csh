#!/bin/csh
foreach i (`cat list`)
set jold=194057
foreach j (`cat list2`)
if(${i} > ${j}) then
set jold=${j} 
endif
end
echo ${i}_${jold}
end
