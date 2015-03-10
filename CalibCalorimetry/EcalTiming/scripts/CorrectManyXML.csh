#!/bin/tcsh -f

set fed = 601
set runnum = $1
set proc = $2
set type = $3
echo "starting with fed" $fed
while ( $fed < 655 )
    echo "working on fed" $fed
    #ProduceRelTimeOffsetFileCMSSWNew sm_${fed}.xml ../data/SM_${fed}_TTPeakPositionFile${type}_Run_${runnum}.${proc}.txt sm_${fed}_${runnum}
    #ProduceRelTimeOffsetFileCMSSWNew sm_${fed}.xml ../data/SM_${fed}_TTPeakPositionFile${type}_Run${runnum}BeamShots.${proc}.txt sm_${fed}_${runnum}
    ProduceRelTimeOffsetFileCMSSWNew sm_${fed}.xml ../data/SM_${fed}_TTPeakPositionFile${type}_BeamShots.${proc}.txt sm_${fed}_${runnum}
    @ fed  = $fed + 1
end


#end of file
