#!/bin/tcsh -f

set fed = 601
set runnum = $1
#set proc = $2
echo "starting with fed" $fed
while ( $fed < 655 )
    echo "working on fed" $fed
    cp sm_${fed}_${runnum}.xml ${runnum}/sm_${fed}.xml 
    @ fed  = $fed + 1
end


#end of file
